echo 0. remove engine files if exist
if [ ! -d "ml-mobileclip" ]; then
    git clone https://github.com/apple/ml-mobileclip
    cd ml-mobileclip
    python -m pip install --upgrade pip setuptools wheel
    python -m pip --version   # verify upgraded pip - v25 seems to work
    python -m pip install -e . 
    cd ..
fi
if [ ! -d "open_clip" ]; then
    git clone https://github.com/mlfoundations/open_clip.git
    cd open_clip
    git apply ../ml-mobileclip/mobileclip2/open_clip_inference_only.patch
    cp -r ../ml-mobileclip/mobileclip2/* ./src/open_clip/
    pip install -e .
    cd ..
fi
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
/usr/src/tensorrt/bin/trtexec --onnx=openclip_image_encoder.onnx --minShapes=image_input:1x3x256x256 --optShapes=image_input:4x3x256x256 --maxShapes=image_input:8x3x256x256 --saveEngine=image_fp16.engine --fp16
echo 5. Create text engine
/usr/src/tensorrt/bin/trtexec --onnx=openclip_text_encoder.onnx --saveEngine=text_fp16.engine --fp16
echo 6. test
python trt_pytorch_cmp.py
echo 7. cleanup "(if passing)"
