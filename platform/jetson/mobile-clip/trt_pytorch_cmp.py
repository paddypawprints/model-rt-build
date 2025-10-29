#!/usr/bin/env python3
"""
trt_pytorch_cmp.py

Small runner for a TensorRT image-encoder engine exported with input (-1,3,256,256).

This revision fixes a bug that caused an attempted allocation with a negative
or nonsensical element count when the engine's declared output shape contained
dynamic dimensions (e.g. (-1, 512)) but the runtime could not call
ctx.set_tensor_shape. The script now constructs concrete output shapes by
replacing dynamic dims with sensible values derived from the concrete input
shape (most commonly the batch dimension), validates element counts, and
raises a clear error if allocation sizes would be invalid.

Requirements:
 - tensorrt, pycuda, torch, open_clip available in the environment
 - engine was exported with input name "image_input" and output name "image_features"
"""

from __future__ import annotations

import time
import numpy as np
from PIL import Image
import traceback
import sys

import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

import torch
import torch.nn.functional as F
import open_clip
from mobileclip.modules.common.mobileone import reparameterize_model

# Hard-coded quick config
ENGINE_PATH = "image_fp16.engine"
IMAGE_PATH = "cat.jpeg"
MODEL_NAME = "MobileCLIP2-S0"
PRETRAINED_PATH = "/home/patrick/mobileclip2_s0.pt"
TEXT_PROMPT = "a photo of a cat"


def replace_dynamic_dims(declared_shape: tuple, concrete_input_shape: tuple) -> tuple:
    """
    Replace dynamic (<=0) dims in declared_shape with values derived from
    concrete_input_shape where possible. If declared and concrete shapes have
    different ranks, map leading dims (batch) first; otherwise fall back to 1.

    Typical use:
      declared_out = (-1, 512)
      concrete_in = (N, 3, 256, 256)
      -> (N, 512)
    """
    out = []
    for i, d in enumerate(declared_shape):
        if isinstance(d, int) and d > 0:
            out.append(int(d))
        else:
            # Prefer to fill from concrete_input_shape at the same index if available
            if i < len(concrete_input_shape):
                out.append(int(concrete_input_shape[i]))
            else:
                # Common fallback: if first declared dim is dynamic and we have at least one in concrete_input_shape,
                # use batch dimension (concrete_input_shape[0]); else default to 1
                if len(concrete_input_shape) > 0:
                    out.append(int(concrete_input_shape[0]))
                else:
                    out.append(1)
    return tuple(out)


def main():
    try:
        # Load engine/runtime/context
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(ENGINE_PATH, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError("Failed to deserialize engine")

        ctx = engine.create_execution_context()

        INPUT_TENSOR_NAME = "image_input"
        OUTPUT_TENSOR_NAME = "image_features"

        # Declared shapes (may contain -1)
        try:
            declared_in_shape = tuple(ctx.get_tensor_shape(INPUT_TENSOR_NAME))
            declared_out_shape = tuple(ctx.get_tensor_shape(OUTPUT_TENSOR_NAME))
        except Exception:
            # If name-based shape queries not available, fall back to placeholders
            declared_in_shape = (-1, 3, 256, 256)
            declared_out_shape = (-1, 512)

        # Try to get dtypes; fall back to float32 if engine doesn't expose the API
        try:
            in_dtype = trt.nptype(engine.get_tensor_dtype(INPUT_TENSOR_NAME))
            out_dtype = trt.nptype(engine.get_tensor_dtype(OUTPUT_TENSOR_NAME))
        except Exception:
            in_dtype = np.float32
            out_dtype = np.float32

        print("TRT engine declared input shape, dtype:", declared_in_shape, in_dtype)
        print("TRT engine declared output shape, dtype:", declared_out_shape, out_dtype)

        # Load model and preprocess
        model_kwargs = {}
        if not (MODEL_NAME.endswith("S3") or MODEL_NAME.endswith("S4") or MODEL_NAME.endswith("L-14")):
            model_kwargs = {"image_mean": (0, 0, 0), "image_std": (1, 1, 1)}

        model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED_PATH, **model_kwargs)
        tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        model.eval()
        model = reparameterize_model(model)

        # Preprocess image to a torch tensor (1,C,H,W)
        pil = Image.open(IMAGE_PATH).convert("RGB")
        torch_input = preprocess(pil).unsqueeze(0)  # (1,C,H,W)

        # If declared spatial dims are concrete (>0) and differ, resize CPU tensor
        try:
            _, _, decl_H, decl_W = declared_in_shape
        except Exception:
            decl_H = decl_W = -1

        b, c, h, w = tuple(torch_input.shape)
        if decl_H > 0 and decl_W > 0 and (h != decl_H or w != decl_W):
            print(f"Resizing preprocessed tensor from ({h},{w}) -> ({decl_H},{decl_W}) to match engine declared spatial dims")
            torch_input = F.interpolate(torch_input, size=(decl_H, decl_W), mode="bilinear", align_corners=False)
            b, c, h, w = tuple(torch_input.shape)

        # Build concrete input shape from actual tensor (replace -1 in declared with actuals)
        concrete_in_shape = []
        declared_in = list(declared_in_shape)
        actual = tuple(torch_input.shape)
        for i in range(len(declared_in)):
            d = declared_in[i]
            if isinstance(d, int) and d > 0:
                concrete_in_shape.append(int(d))
            else:
                # use actual value if available at this axis, otherwise use value from actual[0] (batch)
                if i < len(actual):
                    concrete_in_shape.append(int(actual[i]))
                else:
                    concrete_in_shape.append(int(actual[0]) if len(actual) > 0 else 1)
        concrete_in_shape = tuple(concrete_in_shape)
        print("Concrete input shape to use:", concrete_in_shape)

        # If ctx supports set_tensor_shape, set it; otherwise we'll infer concrete output shape
        if hasattr(ctx, "set_tensor_shape"):
            try:
                ctx.set_tensor_shape(INPUT_TENSOR_NAME, concrete_in_shape)
                # re-query shapes
                in_shape = tuple(ctx.get_tensor_shape(INPUT_TENSOR_NAME))
                out_shape = tuple(ctx.get_tensor_shape(OUTPUT_TENSOR_NAME))
                print("After set_tensor_shape -> in_shape:", in_shape, "out_shape:", out_shape)
            except Exception as e:
                print("Warning: ctx.set_tensor_shape failed:", e)
                in_shape = concrete_in_shape
                # derive concrete output shape by replacing dynamic dims
                out_shape = replace_dynamic_dims(declared_out_shape, in_shape)
                print("Derived out_shape (no set_tensor_shape):", out_shape)
        else:
            in_shape = concrete_in_shape
            out_shape = replace_dynamic_dims(declared_out_shape, in_shape)
            print("ctx.set_tensor_shape not available; using in_shape and derived out_shape:", in_shape, out_shape)

        # Convert tensor to numpy with correct dtype and shape
        np_input = torch_input.cpu().numpy()
        # validate elements
        expected_in_elems = int(np.prod(in_shape))
        if np_input.ravel().size != expected_in_elems:
            raise RuntimeError(f"Preprocessed tensor element count {np_input.ravel().size} != expected {expected_in_elems} for in_shape {in_shape}")

        # Cast to engine dtype
        if in_dtype == np.float16:
            np_input = np_input.astype(np.float16, copy=False)
        else:
            np_input = np_input.astype(np.float32, copy=False)
        np_input = np_input.reshape(in_shape)

        # Prepare host/device buffers with validated positive sizes
        out_elems = int(np.prod(out_shape))
        if expected_in_elems <= 0 or out_elems <= 0:
            raise RuntimeError(f"Invalid allocation size in_elems={expected_in_elems} out_elems={out_elems}")

        host_in = cuda.pagelocked_empty(expected_in_elems, in_dtype)
        host_out = cuda.pagelocked_empty(out_elems, out_dtype)

        # copy into host_in safely
        np.copyto(np.frombuffer(host_in, dtype=in_dtype, count=expected_in_elems), np_input.ravel())

        d_in = cuda.mem_alloc(host_in.nbytes)
        d_out = cuda.mem_alloc(host_out.nbytes)

        # Bind by name (Jetson name-based API expected)
        if not hasattr(ctx, "set_tensor_address"):
            # If set_tensor_address is missing, try the binding-index path (but minimal here)
            raise RuntimeError("ctx.set_tensor_address not available on this TRT build; this script expects Jetson name-based API.")
        ctx.set_tensor_address(INPUT_TENSOR_NAME, int(d_in))
        ctx.set_tensor_address(OUTPUT_TENSOR_NAME, int(d_out))

        stream = cuda.Stream()
        cuda.memcpy_htod_async(d_in, host_in, stream)

        t0 = time.time()
        ctx.execute_async_v3(stream_handle=stream.handle)
        t1 = time.time()

        cuda.memcpy_dtoh_async(host_out, d_out, stream)
        stream.synchronize()
        t2 = time.time()

        trt_out = np.array(host_out).reshape(out_shape)
        if trt_out.dtype == np.float16:
            trt_out = trt_out.astype(np.float32)
        trt_vec = trt_out.ravel().astype(np.float32)

        print(f"TRT exec ms: {(t1-t0)*1000:.2f}, total ms: {(t2-t0)*1000:.2f}")
        print("TRT output shape:", trt_out.shape)

        # Compare with PyTorch model outputs
        with torch.no_grad():
            image_tensor = torch_input
            image_feats = model.encode_image(image_tensor).cpu().numpy().astype(np.float32).ravel()
            tokenized = tokenizer([TEXT_PROMPT])
            text_feats = model.encode_text(tokenized).cpu().numpy().astype(np.float32).ravel()
            image_feats /= np.linalg.norm(image_feats)
            text_feats /= np.linalg.norm(text_feats)

        # Normalize TRT vector
        trt_norm = np.linalg.norm(trt_vec)
        trt_unit = (trt_vec / trt_norm) if trt_norm != 0 else trt_vec

        def cos(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

        print("Norms: TRT", np.linalg.norm(trt_vec), "Torch_image", np.linalg.norm(image_feats), "Torch_text", np.linalg.norm(text_feats))
        print("Cosine TRT vs Torch_image:", cos(trt_vec, image_feats))
        print("Cosine Torch_image vs Torch_text:", cos(image_feats, text_feats))
        print("Cosine TRT vs Torch_text:", cos(trt_vec, text_feats))

        np.save("trt_image_features.npy", trt_vec)
        np.save("torch_image_features.npy", image_feats)
        np.save("torch_text_features.npy", text_feats)
        print("Saved trt_image_features.npy, torch_image_features.npy, torch_text_features.npy")

    except Exception:
        print("Exception encountered:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
