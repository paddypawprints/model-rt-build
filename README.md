# model-rt-build

Repository to build TensorRT engines for multiple models on multiple platforms (Jetson, x86, ...).
Keep each model's build and runtime dependencies isolated in a virtualenv. The repo organizes platform-specific builds under `platform/`.

Quick repo layout
- platform/
  - jetson/
    - mobile-clip/
    - onnxruntime/
    - yolov8n/
      - yolov8n/
- build scripts and per-model tooling live inside each platform/model directory.

Goals
- Provide repeatable, logged build of ONNX -> TensorRT engines.
- Keep per-model Python dependencies isolated (pyproject.toml per model).
- Verify engines by comparing TensorRT outputs with reference PyTorch/OpenCLIP outputs.
- Minimal, debuggable scripts (hard-coded where useful).

Prerequisites
- Host with NVIDIA GPU + CUDA + NVIDIA driver matching TensorRT requirements.
- TensorRT (trtexec available in PATH, example: `/usr/src/tensorrt/bin/trtexec`).
- Python 3.8+ with pip and virtualenv support.
- pycuda (if using the Python runtime tests), tensorrt Python bindings, torch, open_clip, huggingface_hub (as needed per model).
- On Jetson: nvarguscamerasrc / GStreamer for camera use; DeepStream if you want zero-copy video pipelines.
- HF CLI or huggingface_hub for model downloads.

Recommended environment layout per model
- Create a per-model virtualenv (uses system site packages if needed):
  python3 -m venv {model} --system-site-packages
  source {model}/bin/activate
- Install per-model dependencies using the model's pyproject.toml (or requirements.txt).

Typical per-model files you should have
- onnx_export.py         — exports PyTorch model -> ONNX
- build.sh or build_improved.sh — downloads weights, exports ONNX, calls trtexec to build engines, runs verification
- trt_*_compare.py      — runs inference on TRT engine and PyTorch model and compares outputs
- pyproject.toml        — lists Python deps for that model/platform
- README.md (optional)  — model/platform notes

How to build (example flow)
1. Activate the model virtualenv:
   source platform/jetson/mobile-clip/venv/bin/activate

2. Download model weights (recommended: huggingface_hub)
   - Preferred (Python):
     python - <<'PY'
     from huggingface_hub import hf_hub_download
     hf_hub_download(repo_id="apple/MobileCLIP2-S0", filename="mobileclip2_s0.pt", repo_type="model", cache_dir=".")
     PY
   - Or HF CLI:
     hf download apple/MobileCLIP2-S0

3. Export ONNX:
   python onnx_export.py
   - Check the script prints verification messages (e.g. "Image encoder exported successfully").

4. Build TensorRT engines:
   /usr/src/tensorrt/bin/trtexec --onnx=openclip_image_encoder.onnx --saveEngine=image_fp16.engine --fp16
   /usr/src/tensorrt/bin/trtexec --onnx=openclip_text_encoder.onnx  --saveEngine=text_fp16.engine  --fp16

5. Verify:
   python trt_pytorch_cmp.py
   - Confirm cosine similarity / norms are acceptable and files saved (e.g. `trt_image_features.npy`).

Use the provided improved build script
- `build_improved.sh` (or `build_improved.sh` you place in model dir) will:
  - log everything to `build.log`,
  - print concise progress messages,
  - look for success patterns in step output and stop on failure.
- Make executable and run:
  chmod +x build_improved.sh
  ./build_improved.sh

Caching and Hugging Face
- HF cache default: `${HF_HOME:-$HOME/.cache/huggingface/hub}`
- You can copy from cache to working dir:
  find "${HF_HOME:-$HOME/.cache/huggingface/hub}" -type f -name "mobileclip2_s0.pt" -print -quit
- Better: use `hf_hub_download` to download directly to a chosen location.

Logging
- Build scripts should pipe stdout/stderr to a `build.log` file (the included `build_improved.sh` does this).
- Keep the log file with build artifacts for debugging.

Checks and success markers (examples your scripts should look for)
- HF download: lines containing `Fetching` or `100%` or `Downloaded`.
- ONNX export: `Image encoder exported successfully`, `Text encoder exported successfully`, `verified`.
- trtexec engine build: `Engine built`, `Created engine`, `PASSED TensorRT.trtexec`, or final `Created engine with size:`.
- Post-test: presence of `Saved trt_image_features.npy` and `Cosine TRT vs Torch_image`.

Common pitfalls & troubleshooting
- ONNX opset/version mismatch: exporter may pick opset 18. Adjust torch.onnx export opset_version or update ONNX/TensorRT to support opset 18.
- Dynamic shapes: trtexec may override shapes if not provided; fix by providing optimization profile (trtexec flags or use builder APIs).
- Int64 text input binding: ensure text input uses INT64 for token ids (trtexec may warn).
- Memory and time: building engines can take minutes and require RAM/GPU memory; monitor usage.
- If trtexec reports unstable GPU compute time, try locking GPU clocks or add `--useSpinWait`.
- If using Jetson and you need zero-copy video input, use DeepStream or GStreamer NVMM pipelines instead of simple OpenCV capture.

Performance measurement
- Warm up before timing.
- Use CUDA events or `trtexec` built-in measurements.
- Measure kernel-only vs end-to-end (include H2D/D2H as appropriate).

How to add a new model/platform
1. Create new directory under `platform/<platform>/<model>/`.
2. Add `pyproject.toml` describing dependencies.
3. Add `onnx_export.py`, `build.sh` (or use `build_improved.sh`), and verification script.
4. Document in the model's README.md: expected input shapes, HF repo id, known issues.

Example per-model README (minimal)
- Repo id on HF
- Weight filename expected
- ONNX input shapes and opset used
- trtexec invocation lines
- Notes about precision (FP32/FP16) and GPU requirements

CI and automation tips
- CI should run only export/verification on smaller models or use a cached engine artifact due to time and GPU access requirements.
- To automate: have a build machine with the right driver/TensorRT installed and a script that runs `build_improved.sh` and uploads artifacts to an artifact store.

Security and licenses
- Check model licenses before redistribution.
- Do not commit large weight files to the repo. Keep weights in HF or external storage and download during the build.

If you want, I can:
- produce a template `pyproject.toml` for MobileCLIP,
- create a per-model README template you can drop into each model folder,
- or commit the improved build script into `platform/jetson/mobile-clip/`.

