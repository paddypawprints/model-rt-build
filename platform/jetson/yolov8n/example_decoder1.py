#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import argparse
from typing import Any, Tuple, List, Union

import cv2
import numpy as np
import torch

from ultralytics.utils import ASSETS, YAML
from ultralytics.utils.checks import check_yaml

CLASSES = YAML.load(check_yaml("coco8.yaml"))["names"]
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))


def draw_bounding_box(
    img: np.ndarray, class_id: int, confidence: float, x: int, y: int, x_plus_w: int, y_plus_h: int
) -> None:
    """
    Draw bounding boxes on the input image based on the provided arguments.
    """
    label = f"{CLASSES[class_id]} ({confidence:.2f})"
    # ensure BGR int tuple for cv2
    col = colors[class_id]
    color = (int(col[0]), int(col[1]), int(col[2]))
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (max(0, x - 10), max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def letterbox_image(img: np.ndarray, target_size: int = 640, color: Tuple[int, int, int] = (114, 114, 114)):
    """
    Maintain aspect ratio: scale -> resize -> center-pad to (target_size, target_size).
    Returns (canvas, ratio, pad_x, pad_y) where ratio is the resize factor applied
    to the original image (new_w = orig_w * ratio).
    """
    h0, w0 = img.shape[:2]
    r = min(target_size / w0, target_size / h0)
    new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_size, target_size, 3), color, dtype=np.uint8)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, float(r), int(pad_x), int(pad_y)


# --- CV2-style preprocessors (return blob, scale) - kept for quick interchangeability ---


def preprocess_letterbox(input_image: str, size: int = 640) -> Tuple[np.ndarray, float]:
    """
    Read image, letterbox to (size,size) using aspect-ratio-preserving padding,
    and return a cv2.dnn-compatible blob and a scale value compatible with
    the existing code (scale = length/size where length = max(orig_h, orig_w)).

    This function keeps the same (blob, scale) signature as preprocess_cv2 for
    minimal-change testing.
    """
    img = cv2.imread(input_image)
    if img is None:
        raise RuntimeError(f"failed to read image {input_image}")

    h, w = img.shape[:2]
    # build letterbox canvas
    canvas, r, pad_x, pad_y = letterbox_image(img, target_size=size)
    # compute compatibility scale = length/size (same as historical preprocess_cv2)
    length = max(h, w)
    scale = float(length) / float(size)
    blob = cv2.dnn.blobFromImage(canvas, scalefactor=1 / 255.0, size=(size, size), swapRB=True)
    return blob, scale


def preprocess_blob_equivalent(input_image: str, size: int = 640) -> Tuple[np.ndarray, float]:
    """
    Read image, pad to a square canvas (top-left placement like the original example),
    and produce a blob equivalent to cv2.dnn.blobFromImage(canvas, scalefactor=1/255, size=(size,size), swapRB=True).

    Keeps the (blob, scale) signature (scale = length/size) for interchangeability.
    """
    img = cv2.imread(input_image)
    if img is None:
        raise RuntimeError(f"failed to read image {input_image}")

    h, w = img.shape[:2]
    length = max(h, w)
    canvas = np.zeros((length, length, 3), dtype=np.uint8)
    canvas[0:h, 0:w] = img
    scale = float(length) / float(size)
    blob = cv2.dnn.blobFromImage(canvas, scalefactor=1 / 255.0, size=(size, size), swapRB=True)
    return blob, scale


def preprocess_cv2_from_image(img: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Prepare square-padded image and return blob + scale.
    (Used by main when original_image is already loaded.)
    """
    h, w = img.shape[:2]
    length = max(h, w)
    canvas = np.zeros((length, length, 3), np.uint8)
    canvas[0:h, 0:w] = img
    scale = float(length) / 640.0
    blob = cv2.dnn.blobFromImage(canvas, scalefactor=1 / 255.0, size=(640, 640), swapRB=True)
    return blob, scale


# --- PyTorch-style preprocessors (return tensor, ratio, pad_x, pad_y) ---


def preprocess_letterbox_tensor(input_image: str, size: int = 640, device: str = "cpu") -> Tuple[torch.Tensor, float, int, int]:
    """
    PyTorch pipeline variant of letterbox preprocessing.

    Returns:
      tensor: torch.Tensor shape (1,3,size,size) dtype=float32 values in [0,1] on device.
      ratio: float, the resize factor applied to the original image (new_w = orig_w * ratio).
      pad_x, pad_y: integer left/top padding applied to center the resized image.

    This signature (tensor, ratio, pad_x, pad_y) differs from the cv2 blob signature
    but gives the exact information needed to unpad boxes later:
      x_original = (x_model - pad_x) / ratio
    """
    img = cv2.imread(input_image)
    if img is None:
        raise RuntimeError(f"failed to read image {input_image}")

    h, w = img.shape[:2]
    canvas, ratio, pad_x, pad_y = letterbox_image(img, target_size=size)

    # convert to RGB, float32 in [0,1], and to tensor
    arr = canvas[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device).contiguous()
    return tensor, ratio, pad_x, pad_y


def preprocess_blob_equivalent_tensor(input_image: str, size: int = 640, device: str = "cpu") -> Tuple[torch.Tensor, Union[float, Tuple[float, float]], int, int]:
    """
    PyTorch pipeline variant that emulates cv2.dnn.blobFromImage stretch-resize behavior,
    returning (tensor, ratio, pad_x, pad_y).

    For stretch-resize there is no padding (pad_x/pad_y == 0). The ratio returned is
    the per-axis resize factor applied to the original image -> model canvas:
      ratio can be either a float (if original was square) or a tuple (ratio_w, ratio_h).
    Mapping back to original coordinates:
      x_original = x_model / ratio_w
      y_original = y_model / ratio_h

    Returns:
      tensor: torch.Tensor shape (1,3,size,size) dtype=float32 on device
      ratio: float or (ratio_w, ratio_h)
      pad_x, pad_y: integers (both zero for stretch-resize)
    """
    img = cv2.imread(input_image)
    if img is None:
        raise RuntimeError(f"failed to read image {input_image}")

    h, w = img.shape[:2]
    # Resize (stretch) to the target size
    img_resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    arr = img_resized[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device).contiguous()

    # ratio for mapping back: how the original coords were scaled to get to model canvas
    ratio_w = float(size) / float(w)
    ratio_h = float(size) / float(h)
    ratio = (ratio_w, ratio_h) if abs(ratio_w - ratio_h) > 1e-6 else float(ratio_w)
    return tensor, ratio, 0, 0


def main(onnx_model: str, input_image: str, engine: str) -> List[dict]:
    """
    Load ONNX model (or cv2.dnn path) and run detection on input_image, returning detections.
    Minimal example: only the 'example' (cv2.dnn) engine path is implemented here.
    """
    original_image = cv2.imread(input_image)
    if original_image is None:
        raise RuntimeError(f"failed to read image {input_image}")

    scale = 1.0
    outputs = None

    match engine:
        case "example":
            # Load the ONNX model via OpenCV DNN
            model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(onnx_model)
            blob, scale = preprocess_cv2_from_image(original_image)
            model.setInput(blob)
            outputs = model.forward()
        case "onnx":
            # implement ONNXRuntime path if needed
            raise NotImplementedError("onnx engine path not implemented in this example")
        case "trt":
            raise NotImplementedError("trt engine path not implemented in this example")
        case _:
            raise ValueError(f"unknown engine: {engine}")

    # prepare outputs as in Ultralytics example: transpose first entry
    outputs = np.array([cv2.transpose(outputs[0])])
    rows = outputs.shape[1]

    boxes: List = []
    scores: List = []
    class_ids: List = []

    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        # cv2.minMaxLoc expects an image; for 1D array use reshape
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(classes_scores)
        # max_loc may be (x,y) location; for 1D arrays OpenCV returns (x, y) with y as index for the max
        # fall back to numpy argmax if needed
        try:
            maxClassIndex = int(max_loc[1]) if (isinstance(max_loc, tuple) and len(max_loc) >= 2) else int(max_loc)
        except Exception:
            maxClassIndex = int(np.argmax(classes_scores))
        maxScore = float(max_val)
        if maxScore >= 0.25:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                outputs[0][i][2],
                outputs[0][i][3],
            ]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)

    # Apply NMS (OpenCV)
    if len(boxes) == 0:
        result_boxes = []
    else:
        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)
        # normalize to flat list of indices
        if isinstance(result_boxes, (list, tuple, np.ndarray)):
            try:
                result_boxes = np.array(result_boxes).reshape(-1).tolist()
            except Exception:
                result_boxes = [int(x[0]) if isinstance(x, (list, tuple, np.ndarray)) else int(x) for x in result_boxes]
        else:
            result_boxes = []

    detections: List[dict] = []

    for idx in result_boxes:
        idx = int(idx)
        box = boxes[idx]
        detection = {
            "class_id": class_ids[idx],
            "class_name": CLASSES[class_ids[idx]],
            "confidence": scores[idx],
            "box": box,
            "scale": scale,
        }
        detections.append(detection)
        draw_bounding_box(
            original_image,
            class_ids[idx],
            scores[idx],
            round(box[0] * scale),
            round(box[1] * scale),
            round((box[0] + box[2]) * scale),
            round((box[1] + box[3]) * scale),
        )

    # Display result
    cv2.imshow("image", original_image)
    cv2.waitKey(0)
    cv2.imwrite('output_image.png', original_image)
    cv2.destroyAllWindows()

    return detections


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8n.onnx", help="Input your ONNX model.")
    parser.add_argument("--img", default=str(ASSETS / "bus.jpg"), help="Path to input image.")
    parser.add_argument("--engine", default="example", help="example/onnx/trt")
    args = parser.parse_args()
    main(args.model, args.img, args.engine)
