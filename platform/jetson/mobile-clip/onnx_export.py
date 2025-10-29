import torch
import open_clip
import onnxruntime
import numpy as np
from PIL import Image
from mobileclip.modules.common.mobileone import reparameterize_model

model_name = "MobileCLIP2-S0"
model_kwargs = {}
if not (model_name.endswith("S3") or model_name.endswith("S4") or model_name.endswith("L-14")):
    model_kwargs = {"image_mean": (0, 0, 0), "image_std": (1, 1, 1)}

model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained="./mobileclip2_s0.pt", **model_kwargs)
tokenizer = open_clip.get_tokenizer(model_name)

# Model needs to be in eval mode for inference because of batchnorm layers unlike ViTs
model.eval()

# For inference/model exporting purposes, please reparameterize first
model = reparameterize_model(model)

image = preprocess(Image.open("./cat.jpeg").convert("RGB")).unsqueeze(0)
text = tokenizer(["a diagram", "a dog", "a cat"])

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)

# Separate the image and text encoders for export
image_encoder = model.visual
text_encoder = model.text

# Get the resolution from the preprocessor to create a dummy image
#input_resolution = model.visual.image_size
print(f"Input resolution {model.visual.image_size}")
input_resolution = 256

# Create dummy inputs for the image and text encoders
# Dummy image input: (batch_size, channels, height, width)
dummy_image = torch.randn(4, 3, input_resolution, input_resolution)
#dummy_image = image
# Dummy text input: (batch_size, sequence_length)
dummy_text = torch.randint(low=0, high=model.text.context_length, size=(1, model.text.context_length))

# Export the image encoder to ONNX
print("Exporting image encoder...")
torch.onnx.export(
    image_encoder,
    dummy_image,
    f"openclip_image_encoder.onnx",
    input_names=["image_input"],
    output_names=["image_features"],
    dynamic_axes={
        "image_input": {0: "batch_size"},
        "image_features": {0: "batch_size"},
    },
    opset_version=18,
)
print("Image encoder exported successfully.")

# Export the text encoder to ONNX
print("Exporting text encoder...")
torch.onnx.export(
    text_encoder,
    dummy_text,
    f"openclip_text_encoder.onnx",
    input_names=["text_input"],
    output_names=["text_features"],
    dynamic_axes={
        "text_input": {0: "batch_size"},
        "text_features": {0: "batch_size"},
    },
    opset_version=18,
)
print("Text encoder exported successfully.")

# Verification: Run inference with ONNX Runtime and compare outputs
print("Verifying ONNX models...")

# Image encoder verification
ort_sess_img = onnxruntime.InferenceSession("openclip_image_encoder.onnx")
# The dummy tensor needs to be converted to a NumPy array for ONNX Runtime
ort_input_img = {ort_sess_img.get_inputs()[0].name: dummy_image.numpy()}
ort_output_img = ort_sess_img.run(None, ort_input_img)[0]
torch_output_img = image_encoder(dummy_image).detach().numpy()

assert np.allclose(torch_output_img, ort_output_img, atol=1e-5), "Image encoder outputs do not match!"
print("Image encoder verified.")

# Text encoder verification
ort_sess_txt = onnxruntime.InferenceSession("openclip_text_encoder.onnx")
ort_input_txt = {ort_sess_txt.get_inputs()[0].name: dummy_text.numpy()}
ort_output_txt = ort_sess_txt.run(None, ort_input_txt)[0]
torch_output_txt = text_encoder(dummy_text).detach().numpy()

assert np.allclose(torch_output_txt, ort_output_txt, atol=1e-5), "Text encoder outputs do not match!"
print("Text encoder verified.")
