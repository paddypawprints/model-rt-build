# model-rt-build — yolov8n (local, CPU-first)

This README shows a minimal, local workflow to install dependencies, export YOLOv8n -> ONNX, run/verify with onnxruntime (CPU), and optionally build TensorRT engines if you have TensorRT installed. Instructions are intentionally simple and local — no Jetson/CUDA required for the ONNX + verification path.

## Overview
- Create a per-model virtualenv.
- Install CPU-only runtime deps (onnxruntime-cpu).
- Export model to ONNX and run a quick inference with onnxruntime.
- Optionally build TensorRT engines on machines that have TensorRT/CUDA (separate step).

## Prerequisites
- Python 3.8+
- git, curl/wget (optional)
- (Optional, only for TensorRT engine build) NVIDIA drivers, CUDA, TensorRT and `trtexec`.

## Install (local, CPU runtime)
Run from the `platform/jetson/yolov8n/` directory (or project root):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
# Install minimal runtime deps (CPU)
python -m pip install ultralytics onnx onnxruntime-cpu numpy pillow opencv-python-headless
```

Notes:
- `onnxruntime-cpu` is installed to avoid any CUDA/Jetson dependency.
- If you prefer to use the project's `pyproject.toml`, you can `pip install -e .` after activating the venv, then install `onnxruntime-cpu` explicitly to ensure CPU runtime.

## Export YOLOv8n -> ONNX
Put your script (example below) in the model folder and run it.

Example `export_onnx.py` (the simple script you already have):
```python
from ultralytics import YOLO

# Load the YOLOv8n PyTorch model file (yolov8n.pt should be present)
model = YOLO("yolov8n.pt")

# Export the model to ONNX format (produces 'yolov8n.onnx')
model.export(format="onnx")
```

Run:
```bash
python export_onnx.py
```

Expected outcome:
- `yolov8n.onnx` created in the working directory.

## Run inference with ONNX (CPU)
Quick test using the ultralytics ONNX wrapper (or use onnxruntime directly):

Example `run_onnx.py`:
```python
from ultralytics import YOLO

# load the exported ONNX model
onnx_model = YOLO("yolov8n.onnx")

# run inference on a sample image (remote URL or local file)
results = onnx_model("https://ultralytics.com/images/bus.jpg")
print(results)
```

Run:
```bash
python run_onnx.py
```

This uses CPU-only onnxruntime and verifies the exported model runs.

## Optional: Build TensorRT engine (only on machines with TensorRT)
If you have TensorRT and want to build engines, do that with a separate script or CI job (do NOT run as part of pip install). Example `trtexec` command:

```bash
/usr/src/tensorr

