#!/usr/bin/env python3
"""
Compare TensorRT outputs when using different image-preparation methods.

- This is a standalone, hard-coded script. No CLI flags are parsed.
- It runs the TRT engine (hard-coded path) with each of the image-prep variants
  (cv2 blob, padded blob, letterbox blob, and the two tensor->blob wrappers),
  decodes detections using the same simple decode/NMS logic used in the example,
  prints each method's detections (class name + box w x h) and compares them to
  the cv2_blob reference.

Important: This script expects to be run on a system with:
 - TensorRT Python bindings (tensorrt)
 - pycuda
 - OpenCV with Python bindings
 - ultralytics package available (for ASSETS/YAML names)
If any of those are not available the script will raise a clear error.

Save as yolov8_trt_prep_compare.py and run on your Jetson (or TRT-capable system).

Behavior:
 - Hard-coded engine path: "yolov8n_fp16.engine"
 - Hard-coded image path: ASSETS / "bus.jpg" (from ultralytics utils)
 - Hard-coded TRT tensor names: "images" (input), "output0" (output)
 - Uses blob-style signature for preprocessing functions so they are interchangeable.
"""
from __future__ import annotations

import math
from typing import Any, Callable, List, Tuple, Union

import cv2
import numpy as np
import torch

# optional imports
try:
    import tensorrt as trt  # type: ignore
    import pycuda.driver as cuda  # type: ignore
    import pycuda.autoinit  # initializes CUDA context
except Exception:
    trt = None
    cuda = None

try:
    from ultralytics.utils import ASSETS, YAML
    from ultralytics.utils.checks import check_yaml
    CLASSES = YAML.load(check_yaml("coco8.yaml"))["names"]
except Exception:
    # fallback class names if ultralytics not installed
    CLASSES = [str(i) for i in range(80)]

# Hard-coded paths / names (per your request)
ENGINE_PATH = "yolov8n_fp16.engine"
IMAGE_PATH = str(getattr(__import__("pathlib").Path, "cwd")() / "assets" / "bus.jpg") if False else None
# If ultralytics ASSETS is available use it; otherwise expect bus.jpg in the current dir
try:
    from ultralytics.utils import ASSETS  # type: ignore
    IMAGE_PATH = str(ASSETS / "bus.jpg")
except Exception:
    if IMAGE_PATH is None:
        IMAGE_PATH = "bus.jpg"

TRT_INPUT_NAME = "images"
TRT_OUTPUT_NAME = "output0"
MODEL_SIZE = 640
CONF_THRESH = 0.25
NMS_IOU = 0.45


def letterbox_image(img: np.ndarray, target_size: int = 640, color: Tuple[int, int, int] = (114, 114, 114)):
    h0, w0 = img.shape[:2]
    r = min(target_size / w0, target_size / h0)
    new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_size, target_size, 3), color, dtype=np.uint8)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, float(r), int(pad_x), int(pad_y)


# --- Blob-style preprocessors (all share the same signature: (input_image: str, size: int) -> (blob, scale)) ---


def preprocess_cv2(input_image: str, size: int = MODEL_SIZE) -> Tuple[np.ndarray, float]:
    img = cv2.imread(input_image)
    if img is None:
        raise RuntimeError(f"failed to read image {input_image}")
    h, w = img.shape[:2]
    length = max(h, w)
    canvas = np.zeros((length, length, 3), np.uint8)
    canvas[0:h, 0:w] = img
    scale = float(length) / float(size)
    blob = cv2.dnn.blobFromImage(canvas, scalefactor=1.0 / 255.0, size=(size, size), swapRB=True)
    return blob.astype(np.float32), scale


def preprocess_blob_equivalent(input_image: str, size: int = MODEL_SIZE) -> Tuple[np.ndarray, float]:
    img = cv2.imread(input_image)
    if img is None:
        raise RuntimeError(f"failed to read image {input_image}")
    h, w = img.shape[:2]
    length = max(h, w)
    canvas = np.zeros((length, length, 3), dtype=np.uint8)
    canvas[0:h, 0:w] = img
    scale = float(length) / float(size)
    blob = cv2.dnn.blobFromImage(canvas, scalefactor=1.0 / 255.0, size=(size, size), swapRB=True)
    return blob.astype(np.float32), scale


def preprocess_letterbox(input_image: str, size: int = MODEL_SIZE) -> Tuple[np.ndarray, float]:
    img = cv2.imread(input_image)
    if img is None:
        raise RuntimeError(f"failed to read image {input_image}")
    h, w = img.shape[:2]
    canvas, r, pad_x, pad_y = letterbox_image(img, target_size=size)
    length = max(h, w)
    scale = float(length) / float(size)  # keep compatibility scale semantics
    blob = cv2.dnn.blobFromImage(canvas, scalefactor=1.0 / 255.0, size=(size, size), swapRB=True)
    return blob.astype(np.float32), scale


# --- Tensor-style preprocessors converted to blob signature ---


def preprocess_letterbox_tensor_blob(input_image: str, size: int = MODEL_SIZE) -> Tuple[np.ndarray, float]:
    # use CPU tensor conversion then to numpy blob (NCHW)
    img = cv2.imread(input_image)
    if img is None:
        raise RuntimeError(f"failed to read image {input_image}")
    canvas, ratio, pad_x, pad_y = letterbox_image(img, target_size=size)
    arr = canvas[:, :, ::-1].astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    blob = np.ascontiguousarray(tensor.cpu().numpy().astype(np.float32))
    h, w = img.shape[:2]
    length = max(h, w)
    scale = float(length) / float(size)
    return blob, scale


def preprocess_blob_equivalent_tensor_blob(input_image: str, size: int = MODEL_SIZE) -> Tuple[np.ndarray, float]:
    img = cv2.imread(input_image)
    if img is None:
        raise RuntimeError(f"failed to read image {input_image}")
    h, w = img.shape[:2]
    img_resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    arr = img_resized[:, :, ::-1].astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    blob = np.ascontiguousarray(tensor.cpu().numpy().astype(np.float32))
    length = max(h, w)
    scale = float(length) / float(size)
    return blob, scale


# --- TRT runner (name-based API) ---------------------------------------------


def run_trt_engine_named(engine_path: str, input_blob: np.ndarray, input_tensor_name: str, output_tensor_name: str) -> List[np.ndarray]:
    if trt is None or cuda is None:
        raise RuntimeError("TensorRT (tensorrt) and pycuda are required to run TRT engine.")
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize engine from {engine_path}")
    ctx = engine.create_execution_context()

    # Query shapes/dtypes by tensor name (name-based API)
    in_shape = tuple(ctx.get_tensor_shape(input_tensor_name))
    out_shape = tuple(ctx.get_tensor_shape(output_tensor_name))
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

    # Postprocess: squeeze leading batch dim if present and ==1, else error.
    out_np = np.asarray(trt_out)
    print(f"[TRT] raw output shape: {out_np.shape} dtype: {out_np.dtype}")
    if out_np.ndim == 3:
        if out_np.shape[0] == 1:
            out_np = np.squeeze(out_np, axis=0)  # -> (A, P)
            print(f"[TRT] squeezed leading batch dim -> {out_np.shape}")
        else:
            raise RuntimeError(f"TRT output has batch dimension {out_np.shape[0]} > 1; only batch==1 supported here.")
    if out_np.ndim != 2:
        raise RuntimeError(f"TRT output has unexpected rank {out_np.ndim}; expected 2 after squeezing. Shape: {out_np.shape}")
    return [out_np]


# --- helpers: convert engine output to preds (P, A) and decode to boxes/classes ---


def outputs_to_preds_84_85(arr: np.ndarray) -> np.ndarray:
    """
    Normalize an engine output (2D or 1D) into preds (P, A) where A in {84,85}.
    Accepts arr that may be shape (A,P) or (P,A) or 1D flattened.
    """
    a = np.asarray(arr)
    if a.ndim == 1:
        # try reshape by A
        for A in (84, 85):
            if a.size % A == 0:
                preds = a.reshape(-1, A)
                return preds
        raise RuntimeError(f"Cannot reshape flat output of size {a.size} into (*,84/85).")
    if a.ndim == 2:
        # if rows correspond to attributes (A,P) and rows==84/85, transpose to (P,A)
        if a.shape[0] in (84, 85):
            preds = a.T.copy()
            return preds
        # if cols correspond to attributes (P,A)
        if a.shape[1] in (84, 85):
            preds = a.copy()
            return preds
        # otherwise guess: if rows < cols, treat as (P,A) already
        if a.shape[0] < a.shape[1]:
            return a.copy()
        else:
            return a.T.copy()
    raise RuntimeError(f"Unsupported output rank {a.ndim} for normalization.")


def decode_preds_to_detections(preds: np.ndarray, conf_thresh: float = CONF_THRESH) -> List[dict]:
    """
    Simple decode used in the example:
    - preds: (P, A) rows are [cx, cy, w, h, ...class/logits...]
    - For each pred: find best class as argmax over preds[i,4:], use that score as maxScore
    - Keep if maxScore >= conf_thresh.
    - Return boxes in original image coords after scaling is applied by caller.
    """
    P, A = preds.shape
    boxes = []
    scores = []
    class_ids = []
    for i in range(P):
        row = preds[i]
        classes_scores = row[4:]
        # find max class score and index
        # Use numpy argmax directly (this mirrors cv2.minMaxLoc usage earlier)
        max_idx = int(np.argmax(classes_scores))
        max_score = float(classes_scores[max_idx])
        if max_score >= conf_thresh:
            cx, cy, w, h = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            left = cx - 0.5 * w
            top = cy - 0.5 * h
            boxes.append([left, top, w, h])
            scores.append(max_score)
            class_ids.append(max_idx)
    # perform NMS using OpenCV dnn NMSBoxes
    if len(boxes) == 0:
        return []
    try:
        keep = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, NMS_IOU, 0.5)
        # normalize to list of indices
        if isinstance(keep, (list, tuple, np.ndarray)):
            try:
                keep = np.array(keep).reshape(-1).tolist()
            except Exception:
                keep = [int(x[0]) if isinstance(x, (list, tuple, np.ndarray)) else int(x) for x in keep]
        else:
            keep = []
    except Exception:
        # fallback: no NMS or unexpected return type
        keep = list(range(len(boxes)))
    detections = []
    for idx in keep:
        idx = int(idx)
        l, t, w, h = boxes[idx]
        detections.append(
            {
                "class_id": class_ids[idx],
                "class_name": CLASSES[class_ids[idx]] if class_ids[idx] < len(CLASSES) else str(class_ids[idx]),
                "score": scores[idx],
                "box_xywh": [l, t, w, h],
            }
        )
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


# --- main: run TRT for each preprocessor and compare to cv2_blob reference ---


def run_and_decode_for_blob(blob: np.ndarray, scale: float) -> List[dict]:
    """
    Run TRT engine on blob and decode detections returned scaled to original image coordinates.
    """
    outs = run_trt_engine_named(ENGINE_PATH, blob, TRT_INPUT_NAME, TRT_OUTPUT_NAME)
    out0 = outs[0]  # 2D array (A, P) expected after run_trt_engine_named postprocess
    preds = outputs_to_preds_84_85(out0)  # (P, A)
    detections = decode_preds_to_detections(preds, conf_thresh=CONF_THRESH)
    # map boxes (which are in model canvas coordinates) back to original image using scale
    for det in detections:
        l, t, w, h = det["box_xywh"]
        det["box_xywh_orig"] = [l * scale, t * scale, w * scale, h * scale]
        det["box_xyxy_orig"] = xywh_to_xyxy(det["box_xywh_orig"])
    return detections


def main():
    print("Preparing hard-coded test run: TRT engine path =", ENGINE_PATH)
    preps: List[Tuple[str, Callable[[str, int], Tuple[np.ndarray, float]]]] = [
        ("cv2_blob", preprocess_cv2),
        ("blob_equivalent", preprocess_blob_equivalent),
        ("letterbox_blob", preprocess_letterbox),
        ("blob_equivalent_tensor_blob", preprocess_blob_equivalent_tensor_blob),
        ("letterbox_tensor_blob", preprocess_letterbox_tensor_blob),
    ]

    # Run the reference using cv2_blob first
    print("\n=== Running reference (cv2_blob) ===")
    ref_blob, ref_scale = preprocess_cv2(IMAGE_PATH, MODEL_SIZE)
    print(f"[REF] blob.shape={ref_blob.shape} dtype={ref_blob.dtype} scale={ref_scale}")
    ref_dets = run_and_decode_for_blob(ref_blob, ref_scale)
    print(f"[REF] detections: {len(ref_dets)}")
    for d in ref_dets:
        name = d["class_name"]
        l, t, w, h = d["box_xywh_orig"]
        print(f"  - {name} score={d['score']:.4f} box_wxh={w:.1f}x{h:.1f} at ({l:.1f},{t:.1f})")

    # Run all other preprocessors and compare
    for name, fn in preps:
        print(f"\n=== Running preprocessor: {name} ===")
        try:
            blob, scale = fn(IMAGE_PATH, MODEL_SIZE)
            print(f"[{name}] blob.shape={getattr(blob, 'shape', None)} dtype={getattr(blob, 'dtype', None)} scale={scale}")
            dets = run_and_decode_for_blob(blob, scale)
            print(f"[{name}] detections: {len(dets)}")
            for d in dets:
                name_c = d["class_name"]
                l, t, w, h = d["box_xywh_orig"]
                print(f"  - {name_c} score={d['score']:.4f} box_wxh={w:.1f}x{h:.1f} at ({l:.1f},{t:.1f})")

            # Compare to reference: for each detection in current set find best matching ref detection by IoU
            matches = 0
            for d in dets:
                best_iou = 0.0
                best_ref = None
                for rd in ref_dets:
                    i = iou_xyxy(d["box_xyxy_orig"], rd["box_xyxy_orig"])
                    if i > best_iou:
                        best_iou = i
                        best_ref = rd
                same_class = best_ref is not None and best_ref["class_name"] == d["class_name"]
                print(f"    -> best match IoU={best_iou:.3f} class_match={same_class} (ref_class={best_ref['class_name'] if best_ref else None})")
                if best_iou > 0.5 and same_class:
                    matches += 1
            print(f"[{name}] matched {matches}/{len(dets)} detections to reference with IoU>0.5 and same class.")
        except Exception as e:
            print(f"[{name}] ERROR: {e}")

    print("\nDone. Note: This script must be run on the target device to actually execute TRT engine runs.")


if __name__ == "__main__":
    main()