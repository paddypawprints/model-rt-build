#!/usr/bin/env python3
"""
Compare TensorRT outputs and decoded detections for two preprocess methods only:
 - cv2_blob (pad-to-square, cv2.dnn.blobFromImage)  [reference]
 - blob_equivalent_tensor_blob (test) — uses NumPy + PyTorch operations
   that run on CPU (forced).

This is a hard-coded, single-purpose script intended to run on the target device
with TensorRT + pycuda available. It prints detailed diagnostics comparing:

 1) numerical differences between the two input blobs (shape, dtype, L2, max abs)
 2) first 16 blob values for quick manual inspection
 3) raw TRT engine output differences (L2, max abs, cosine similarity)
 4) first prediction vector (preds[0,:16]) differences
 5) final decoded detections and IoU/class-match comparison

Note: this version fixes a TensorRT API mismatch that caused:
  AttributeError: 'ICudaEngine' object has no attribute 'get_binding_dtype'
We now use the name-based API consistently: ctx.get_tensor_shape(name) +
engine.get_tensor_dtype(name).
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple, Callable, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# optional imports for TensorRT (TRT) runtime
try:
    import tensorrt as trt  # type: ignore
    import pycuda.driver as cuda  # type: ignore
    import pycuda.autoinit  # initializes CUDA context
except Exception:
    trt = None
    cuda = None

# try to use ultralytics assets and coco names if available
try:
    from ultralytics.utils import ASSETS, YAML
    from ultralytics.utils.checks import check_yaml

    CLASSES = YAML.load(check_yaml("coco8.yaml"))["names"]
    IMAGE_PATH = str(ASSETS / "bus.jpg")
except Exception:
    CLASSES = [str(i) for i in range(80)]
    IMAGE_PATH = "bus.jpg"

# Hard-coded parameters / paths
ENGINE_PATH = "yolov8n_fp16.engine"
TRT_INPUT_NAME = "images"
TRT_OUTPUT_NAME = "output0"
MODEL_SIZE = 640
CONF_THRESH = 0.25
NMS_IOU = 0.45


# ----------------------------
# Preprocessing helper methods
# ----------------------------

def letterbox_image(img: np.ndarray, target_size: int = MODEL_SIZE, color: Tuple[int, int, int] = (114, 114, 114)):
    h0, w0 = img.shape[:2]
    r = min(target_size / w0, target_size / h0)
    new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_size, target_size, 3), color, dtype=np.uint8)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, float(r), int(pad_x), int(pad_y)


def preprocess_cv2(input_image: str, size: int = MODEL_SIZE) -> Tuple[np.ndarray, float, Dict]:
    """
    Pad top-left to square canvas, blobFromImage, returns meta with mode pad_to_square.
    (Reference pipeline)
    """
    img = cv2.imread(input_image)
    if img is None:
        raise RuntimeError(f"failed to read image {input_image}")
    h, w = img.shape[:2]
    length = max(h, w)
    canvas = np.zeros((length, length, 3), np.uint8)
    canvas[0:h, 0:w] = img
    scale = float(length) / float(size)
    # Use swapRB=True and scalefactor=1/255 to match training preprocessing used by many models
    blob = cv2.dnn.blobFromImage(canvas, scalefactor=1.0 / 255.0, size=(size, size), swapRB=True)
    meta = {"mode": "pad_to_square", "ratio": None, "pad_x": 0, "pad_y": 0}
    return np.ascontiguousarray(blob.astype(np.float32)), scale, meta


def preprocess_blob_equivalent_tensor_blob(input_image: str, size: int = MODEL_SIZE) -> Tuple[np.ndarray, float, Dict]:
    """
    Stretch-resize to (size,size) using PyTorch on CPU and produce
    a contiguous NCHW float32 numpy blob compatible with TRT runner.

    Requirements satisfied:
      - Uses cv2.imread only to load the image (BGR).
      - Uses numpy + PyTorch ops for resizing, forced to CPU to avoid CUDA kernel issues.
      - Produces channel ordering and normalization equivalent to cv2.dnn.blobFromImage
        with swapRB=True and scalefactor=1/255 (BGR->RGB then /255).
    """
    img = cv2.imread(input_image)
    if img is None:
        raise RuntimeError(f"failed to read image {input_image}")

    h, w = img.shape[:2]

    # Convert BGR -> RGB and normalize to [0,1] using NumPy
    arr = img[:, :, ::-1].astype(np.float32) / 255.0  # now RGB, HWC, float32

    # Force CPU usage for PyTorch operations
    device = torch.device("cpu")
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)  # (1,3,H,W) on CPU

    # Resize on CPU using PyTorch (bilinear interpolation)
    tensor_resized = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)

    # Ensure float32, keep on CPU and produce contiguous numpy NCHW array
    tensor_resized = tensor_resized.to(torch.float32).contiguous()
    blob = np.ascontiguousarray(tensor_resized.cpu().numpy())

    # ratio/scale metadata (stretch semantics)
    ratio_w = float(size) / float(w)
    ratio_h = float(size) / float(h)
    ratio = (ratio_w, ratio_h) if abs(ratio_w - ratio_h) > 1e-6 else float(ratio_w)
    scale = float(max(h, w)) / float(size)
    meta = {"mode": "stretch", "ratio": ratio, "pad_x": 0, "pad_y": 0}
    return blob, scale, meta


# ----------------------------
# TRT runner (name-based API)
# ----------------------------

def run_trt_engine_named(engine_path: str, input_blob: np.ndarray, input_tensor_name: str, output_tensor_name: str):
    if trt is None or cuda is None:
        raise RuntimeError("TensorRT (tensorrt) and pycuda are required to run TRT engine.")
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize engine from {engine_path}")
    ctx = engine.create_execution_context()

    # Use name-based tensor API for shapes/dtypes
    # ctx.get_tensor_shape(name) returns the shape; use engine.get_tensor_dtype(name) for dtype
    in_shape = tuple(ctx.get_tensor_shape(input_tensor_name))
    out_shape = tuple(ctx.get_tensor_shape(output_tensor_name))

    # NOTE: use get_tensor_dtype when using name-based API
    in_dtype = trt.nptype(engine.get_tensor_dtype(input_tensor_name))
    out_dtype = trt.nptype(engine.get_tensor_dtype(output_tensor_name))

    print(f"[TRT] input tensor name {input_tensor_name} shape {in_shape} dtype {in_dtype}")
    print(f"[TRT] output tensor name {output_tensor_name} shape {out_shape} dtype {out_dtype}")

    in_elems = int(np.prod(in_shape))
    out_elems = int(np.prod(out_shape))

    host_in = cuda.pagelocked_empty(in_elems, in_dtype)
    host_out = cuda.pagelocked_empty(out_elems, out_dtype)

    src = np.ascontiguousarray(input_blob).ravel().astype(in_dtype, copy=False)
    if src.size != in_elems:
        raise RuntimeError(f"Input size mismatch: engine expects {in_elems} elements, got {src.size}")
    np.copyto(np.frombuffer(host_in, dtype=in_dtype, count=in_elems), src)

    d_in = cuda.mem_alloc(host_in.nbytes)
    d_out = cuda.mem_alloc(host_out.nbytes)

    ctx.set_tensor_address(input_tensor_name, int(d_in))
    ctx.set_tensor_address(output_tensor_name, int(d_out))

    stream = cuda.Stream()
    cuda.memcpy_htod_async(d_in, host_in, stream)
    ctx.execute_async_v3(stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_out, d_out, stream)
    stream.synchronize()

    trt_out = np.array(host_out).reshape(out_shape)
    if trt_out.dtype == np.float16:
        trt_out = trt_out.astype(np.float32)

    out_np = np.asarray(trt_out)
    print(f"[TRT] raw output shape: {out_np.shape} dtype: {out_np.dtype}")
    if out_np.ndim == 3:
        if out_np.shape[0] == 1:
            out_np = np.squeeze(out_np, axis=0)
            print(f"[TRT] squeezed leading batch dim -> {out_np.shape}")
        else:
            raise RuntimeError("TRT output has batch dimension > 1; only batch==1 supported here.")
    if out_np.ndim != 2:
        raise RuntimeError(f"TRT output has unexpected rank {out_np.ndim}; expected 2 after squeezing.")
    return [out_np]


# ----------------------------
# Output normalization / decode
# ----------------------------

def outputs_to_preds_84_85(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 1:
        for A in (84, 85):
            if a.size % A == 0:
                return a.reshape(-1, A)
        raise RuntimeError(f"Cannot reshape flat output of size {a.size} into (*,84/85).")
    if a.ndim == 2:
        if a.shape[0] in (84, 85):
            return a.T.copy()
        if a.shape[1] in (84, 85):
            return a.copy()
        if a.shape[0] < a.shape[1]:
            return a.copy()
        return a.T.copy()
    raise RuntimeError(f"Unsupported output rank {a.ndim} for normalization.")


def decode_preds_to_detections(preds: np.ndarray, conf_thresh: float = CONF_THRESH):
    P, A = preds.shape
    boxes = []
    scores = []
    class_ids = []
    for i in range(P):
        row = preds[i]
        classes_scores = row[4:]
        max_idx = int(np.argmax(classes_scores))
        max_score = float(classes_scores[max_idx])
        if max_score >= conf_thresh:
            cx, cy, w, h = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            left = cx - 0.5 * w
            top = cy - 0.5 * h
            boxes.append([left, top, w, h])
            scores.append(max_score)
            class_ids.append(max_idx)
    if len(boxes) == 0:
        return []
    try:
        keep = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, NMS_IOU, 0.5)
        if isinstance(keep, (list, tuple, np.ndarray)):
            try:
                keep = np.array(keep).reshape(-1).tolist()
            except Exception:
                keep = [int(x[0]) if isinstance(x, (list, tuple, np.ndarray)) else int(x) for x in keep]
        else:
            keep = []
    except Exception:
        keep = list(range(len(boxes)))
    detections = []
    for idx in keep:
        idx = int(idx)
        l, t, w, h = boxes[idx]
        detections.append({"class_id": class_ids[idx], "class_name": CLASSES[class_ids[idx]] if class_ids[idx] < len(CLASSES) else str(class_ids[idx]), "score": scores[idx], "box_xywh": [l, t, w, h]})
    return detections


def xywh_to_xyxy(box_xywh):
    l, t, w, h = box_xywh
    return [l, t, l + w, t + h]


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-12
    return inter / union


# ----------------------------
# Diagnostics & comparison
# ----------------------------

def compare_blobs(blob_ref: np.ndarray, blob_test: np.ndarray) -> None:
    # flatten both
    a = np.ascontiguousarray(blob_ref).ravel().astype(np.float32)
    b = np.ascontiguousarray(blob_test).ravel().astype(np.float32)
    if a.size != b.size:
        print(f"[DIAG-BLOB] different flattened sizes: ref={a.size} test={b.size}")
    n = min(a.size, b.size)
    diff = a[:n] - b[:n]
    l2 = float(np.linalg.norm(diff))
    maxabs = float(np.max(np.abs(diff)))
    meanabs = float(np.mean(np.abs(diff)))
    frac_gt_1e_6 = float(np.mean(np.abs(diff) > 1e-6))
    cos = float(np.dot(a[:n], b[:n]) / (np.linalg.norm(a[:n]) * np.linalg.norm(b[:n]) + 1e-12))
    print(f"[DIAG-BLOB] L2 diff={l2:.6f} maxabs={maxabs:.6f} meanabs={meanabs:.6f} frac>|1e-6|={frac_gt_1e_6:.6f} cosine={cos:.6f}")
    print("[DIAG-BLOB] ref first16:", a[:16].tolist())
    print("[DIAG-BLOB] test first16:", b[:16].tolist())


def compare_raw_outputs(raw_ref: np.ndarray, raw_test: np.ndarray) -> None:
    a = np.ascontiguousarray(raw_ref).ravel().astype(np.float32)
    b = np.ascontiguousarray(raw_test).ravel().astype(np.float32)
    n = min(a.size, b.size)
    diff = a[:n] - b[:n]
    l2 = float(np.linalg.norm(diff))
    maxabs = float(np.max(np.abs(diff)))
    meanabs = float(np.mean(np.abs(diff)))
    cos = float(np.dot(a[:n], b[:n]) / (np.linalg.norm(a[:n]) * np.linalg.norm(b[:n]) + 1e-12))
    print(f"[DIAG-RAW] L2 diff={l2:.6f} maxabs={maxabs:.6f} meanabs={meanabs:.6f} cosine={cos:.6f}")
    print("[DIAG-RAW] first 32 ref:", a[:32].tolist())
    print("[DIAG-RAW] first 32 test:", b[:32].tolist())
    print("[DIAG-RAW] first 32 diff:", (a[:32] - b[:32]).tolist())


# ----------------------------
# Run & decode for a blob/meta
# ----------------------------

def run_and_decode_for_blob(blob: np.ndarray, scale: float, meta: Dict):
    # Diagnostics of blob
    print("[DIAG] blob shape:", getattr(blob, "shape", None), "dtype:", getattr(blob, "dtype", None), "contiguous:", np.ascontiguousarray(blob).flags["C_CONTIGUOUS"])
    flat = np.asarray(blob).ravel()
    print("[DIAG] first 16 blob values:", flat[:16].tolist())

    outs = run_trt_engine_named(ENGINE_PATH, blob, TRT_INPUT_NAME, TRT_OUTPUT_NAME)
    out0 = outs[0]
    print("[DIAG] engine post-squeeze output shape:", out0.shape, "dtype:", out0.dtype)
    preds = outputs_to_preds_84_85(out0)  # (P,A)
    print("[DIAG] preds.shape:", preds.shape)
    print("[DIAG] first pred vector (raw):", preds[0, :16].tolist())
    dets = decode_preds_to_detections(preds, conf_thresh=CONF_THRESH)
    # Unmap boxes to original coords based on meta
    for det in dets:
        l, t, w, h = det["box_xywh"]
        if meta["mode"] == "letterbox":
            ratio = float(meta["ratio"])
            pad_x = int(meta["pad_x"])
            pad_y = int(meta["pad_y"])
            x_orig = (l - pad_x) / ratio
            y_orig = (t - pad_y) / ratio
            w_orig = w / ratio
            h_orig = h / ratio
        elif meta["mode"] == "pad_to_square":
            x_orig = l * scale
            y_orig = t * scale
            w_orig = w * scale
            h_orig = h * scale
        elif meta["mode"] == "stretch":
            ratio = meta["ratio"]
            if isinstance(ratio, tuple):
                ratio_w, ratio_h = ratio
            else:
                ratio_w = ratio_h = float(ratio)
            x_orig = l / ratio_w
            y_orig = t / ratio_h
            w_orig = w / ratio_w
            h_orig = h / ratio_h
        else:
            x_orig = l * scale
            y_orig = t * scale
            w_orig = w * scale
            h_orig = h * scale
        det["box_xywh_orig"] = [x_orig, y_orig, w_orig, h_orig]
        det["box_xyxy_orig"] = xywh_to_xyxy(det["box_xywh_orig"])
    return dets, preds, out0


# ----------------------------
# Main comparison flow (hard-coded)
# ----------------------------

def main():
    print("Hard-coded run comparing cv2_blob (reference) vs blob_equivalent_tensor_blob (test)")
    # Prepare both blobs
    ref_blob, ref_scale, ref_meta = preprocess_cv2(IMAGE_PATH, MODEL_SIZE)
    test_blob, test_scale, test_meta = preprocess_blob_equivalent_tensor_blob(IMAGE_PATH, MODEL_SIZE)

    print("\n=== Blob diagnostics ===")
    print("[REF] blob.shape=", ref_blob.shape, "dtype=", ref_blob.dtype, "meta=", ref_meta, "scale=", ref_scale)
    print("[TEST] blob.shape=", test_blob.shape, "dtype=", test_blob.dtype, "meta=", test_meta, "scale=", test_scale)
    compare_blobs(ref_blob, test_blob)

    # Run TRT on both (raw outputs)
    print("\n=== Running TRT with reference blob ===")
    ref_dets, ref_preds, ref_raw = run_and_decode_for_blob(ref_blob, ref_scale, ref_meta)

    print("\n=== Running TRT with test blob ===")
    test_dets, test_preds, test_raw = run_and_decode_for_blob(test_blob, test_scale, test_meta)

    # Compare raw outputs (post-squeeze arrays returned by run_trt_engine_named)
    print("\n=== Raw output comparison ===")
    compare_raw_outputs(ref_raw, test_raw)

    # Compare first pred vector differences
    print("\n=== First prediction vector comparison ===")
    p0 = ref_preds[0, :16].astype(np.float32)
    p1 = test_preds[0, :16].astype(np.float32)
    diff = p0 - p1
    print("ref first16:", p0.tolist())
    print("test first16:", p1.tolist())
    print("diff first16:", diff.tolist())
    print("L2 diff pred0:", float(np.linalg.norm(diff)), "maxabs:", float(np.max(np.abs(diff))))

    # Decode and compare detection sets by IoU and class
    print("\n=== Decoded detection comparison ===")
    print(f"Reference detections: {len(ref_dets)}")
    for d in ref_dets:
        name = d["class_name"]
        l, t, w, h = d["box_xywh_orig"]
        print(f"  - {name} score={d['score']:.4f} box_wxh={w:.1f}x{h:.1f} at ({l:.1f},{t:.1f})")
    print(f"Test detections: {len(test_dets)}")
    for d in test_dets:
        name = d["class_name"]
        l, t, w, h = d["box_xywh_orig"]
        print(f"  - {name} score={d['score']:.4f} box_wxh={w:.1f}x{h:.1f} at ({l:.1f},{t:.1f})")

    # Perform matching
    matches = 0
    for td in test_dets:
        best_iou = 0.0
        best_ref = None
        for rd in ref_dets:
            i = iou_xyxy(td["box_xyxy_orig"], rd["box_xyxy_orig"])
            if i > best_iou:
                best_iou = i
                best_ref = rd
        same_class = best_ref is not None and best_ref["class_name"] == td["class_name"]
        print(f"Test det class={td['class_name']} best IoU={best_iou:.3f} class_match={same_class} (ref_class={best_ref['class_name'] if best_ref else None})")
        if best_iou > 0.5 and same_class:
            matches += 1
    print(f"Matched {matches}/{len(test_dets)} test detections to reference with IoU>0.5 and same class.")

    print("\nDone. Review the diagnostics above to determine whether preprocessing or unpad mapping is the root cause.")


if __name__ == "__main__":
    main()
