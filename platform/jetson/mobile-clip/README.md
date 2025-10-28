# Mobile CLIP (Jetson) — README

WARNING — UNDER DEVELOPMENT
This repository and the provided build workflow are still under active development. The scripts work for the environment where they were created but may require debugging, especially the Python environment, platform-specific wheels (PyTorch / ONNX runtimes), and TensorRT invocation. Expect to troubleshoot dependency versions, wheel compatibility, and TensorRT tool paths (trtexec). Use a fresh virtual environment for experiments.

Summary
- Purpose: build and test MobileCLIP models for a Jetson/TensorRT environment.
- What the scripts do (high level):
  - clone/prepare necessary repos (ml-mobileclip, open_clip)
  - fetch MobileCLIP weights from Hugging Face
  - export ONNX models
  - build TensorRT engines (image + text)
  - run a comparison/test script
- Important: Many steps rely on platform-specific wheels, external tools (trtexec), and the correct Python environment. Read the Troubleshooting section if things fail.

Prerequisites
- Linux machine (the build was developed for NVIDIA Jetson style platforms).
- TensorRT installed (trtexec available). Default script uses `/usr/src/tensorrt/bin/trtexec`.
- Python 3.10+ (the environment used here is Python 3.10.12).
- A Python virtual environment (highly recommended).
- Hugging Face CLI / `hf` or ability to download model files from the Hugging Face Hub and place them in the cache.
- Sufficient disk space for model files and caches.
- Optionally: build tools and compilers if you will build ONNX runtime or other native packages.

Quick start — recommended (manual)
1. Create and activate a virtual environment:
   - Linux/macOS:
     ```
     python -m venv .venv
     source .venv/bin/activate
     ```
   - Upgrade pip/setuptools/wheel inside the venv (recommended):
     ```
     python -m pip install --upgrade pip setuptools wheel
     python -m pip --version   # verify upgraded pip (newer pip versions are less likely to hit resolver bugs)
     ```

2. Run the build steps (see the build script below). The script is provided as a reference — read it before running.

Build script (reference)
```bash
# 0. remove engine files if exist (manual step recommended)
echo 0. remove engine files if exist

# clone and install ml-mobileclip (editable)
if [ ! -d "ml-mobileclip" ]; then
    git clone https://github.com/apple/ml-mobileclip
    cd ml-mobileclip
    python -m pip install --upgrade pip setuptools wheel
    python -m pip --version   # verify upgraded pip - v25 seems to work
    python -m pip install -e . 
fi

# clone and patch open_clip
if [ ! -d "open_clip" ]; then
    git clone https://github.com/mlfoundations/open_clip.git
    cd open_clip
    git apply ../ml-mobileclip/mobileclip2/open_clip_inference_only.patch
    cp -r ../ml-mobileclip/mobileclip2/* ./src/open_clip/
    pip install -e .
    cd ..
fi

# install pytorch-image-models (timm)
pip install git+https://github.com/huggingface/pytorch-image-models

echo 1.download from hf
# improvement - test if the file is there first
if [ ! -f "$HOME/.cache/huggingface/hub/models--apple--MobileCLIP2-S0/snapshots/3136ea51c8ed56b9f9abfab04cb816735aaad6cb/mobileclip2_s0.pt" ]; then
    hf download  apple/MobileCLIP2-S0
fi

echo 2.copy from hf cache location
cp $HOME/.cache/huggingface/hub/models--apple--MobileCLIP2-S0/snapshots/3136ea51c8ed56b9f9abfab04cb816735aaad6cb/mobileclip2_s0.pt .

echo 3. extract 32 bit onnx files
python onnx_export.py
if [ ! -f "openclip_image_encoder.onnx" ]; then
    exit 1
fi

echo 4. create image engine
/usr/src/tensorrt/bin/trtexec --onnx=openclip_image_encoder.onnx --saveEngine=image_fp16.engine --fp16

echo 5. Create text engine
/usr/src/tensorrt/bin/trtexec --onnx=openclip_text_encoder.onnx --saveEngine=text_fp16.engine --fp16

echo 6. test
python trt_pytorch_cmp.py

echo 7. cleanup "(if passing)"
```

Notes, important caveats and tips
- Do not run the build script without understanding each step. The script performs installs and uses platform-specific wheels and binaries.
- Hugging Face model download:
  - The script expects the model snapshot in a particular cache path. If you use a different method to download the model, adjust the `cp` path accordingly.
  - The `hf download` command assumes the Hugging Face Hub CLI is installed and authenticated (or that you have access to public model artifacts).
- ONNX export:
  - `onnx_export.py` must exist and successfully export both `openclip_image_encoder.onnx` and `openclip_text_encoder.onnx`.
  - The export may require specific versions of PyTorch/ONNX/onnxruntime; pin versions in your environment if necessary.
- TensorRT engine creation:
  - `trtexec` path may differ by installation. On some systems it is in `/usr/bin` or `/usr/local/TensorRT/bin`. Update the path accordingly.
  - Building engines may require custom flags depending on your hardware (workspace size, explicit batch, int8/fp16 choices).
- Jetson-specific notes:
  - Pre-built PyTorch wheels must match the target Jetson CUDA and architecture.
  - `torch` and `torchvision` are often installed from a special index or local wheel (the script in earlier work used `--index-url https://download.pytorch.org/whl/cu126`).
  - ONNX Runtime GPU wheels for AArch64 were installed from local builds in the work history; if you use those, ensure they are compatible.

Troubleshooting pointers
- If pip fails with resolver errors (AssertionError in resolvelib):
  - Upgrade pip/setuptools/wheel in the venv first: `python -m pip install --upgrade pip setuptools wheel`
  - Try `python -m pip install -e . --no-deps` then install the dependencies one-by-one to find the conflicting package.
  - Alternatively, use a clean venv and install there.
- If trtexec fails or is not found:
  - Check your TensorRT installation and update the `trtexec` path in the script.
  - Check GPU drivers and CUDA compatibility with TensorRT.
- If ONNX export fails:
  - Check PyTorch and ONNX versions; some ops require newer/older ONNX operator sets.
  - Look at the exported model with `onnx.checker.check_model()` to find issues.
- If model inference results mismatch between PyTorch and TRT:
  - Verify input preprocessing is identical for both paths (channel order, normalization, letterbox vs stretch).
  - Use the diagnostic script in this repo (if present) to compare raw outputs and boxes.

Recommended first actions for a fresh developer
1. Create a new venv (see Quick start).
2. Install minimal dependencies (pip, wheel, setuptools).
3. Manually run the individual build steps (clone repo(s), pip install -e ., export ONNX, run trtexec) and fix issues as they appear rather than running the whole script at once.
4. Keep notes of the exact wheel files and versions that work on your machine (especially for torch, torchvision, onnxruntime, pycuda).

Where to get help
- Collect `pip -vvv` logs when installs fail and inspect the last 200 lines for the failing requirement.
- When TensorRT engine build fails, capture the full `trtexec` output and the engine creation flags used.
- If you need help debugging a specific error, include:
  - the failing command and its full output,
  - `python -m pip freeze` (environment),
  - path to `trtexec` and TensorRT version.

Acknowledgements
- This project references and depends on:
  - Apple / ml-mobileclip
  - ml-foundations / open_clip
  - Hugging Face Hub and model artifacts
  - NVIDIA TensorRT and Jetson platform tools

License
- Check upstream project licenses (Apple MIT or project-specific) before redistribution.
