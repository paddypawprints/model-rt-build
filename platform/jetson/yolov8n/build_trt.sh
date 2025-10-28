#!/usr/bin/env bash
# Build a TensorRT engine from an ONNX model using trtexec.
# Usage: ./build_trt.sh [onnx_path] [engine_path] [log_path]
set -euo pipefail

python yolov8n.py

python extract_coco_names.py

ONNX="${1:-yolov8n.onnx}"
ENGINE="${2:-${ONNX%.onnx}_fp16.engine}"
LOG="${3:-build_trt.log}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"

echo "Build started: $(date -u +"%Y-%m-%dT%H:%M:%SZ")" | tee "$LOG"
echo "ONNX: $ONNX" | tee -a "$LOG"
echo "Engine: $ENGINE" | tee -a "$LOG"
echo "trtexec: $TRTEXEC" | tee -a "$LOG"

# Run trtexec and capture stdout/stderr to log
"$TRTEXEC" --onnx="$ONNX" --saveEngine="$ENGINE" --fp16 2>&1 | tee -a "$LOG"

# Look for success markers
if grep -q -E "PASSED TensorRT.trtexec|Engine built|Created engine" "$LOG"; then
  echo "OK: Engine build appears successful." | tee -a "$LOG"
  echo "Build finished: $(date -u +"%Y-%m-%dT%H:%M:%SZ")" | tee -a "$LOG"
  exit 0
else
  echo "ERROR: Engine build failed. See $LOG for details." | tee -a "$LOG"
  exit 1
fi

