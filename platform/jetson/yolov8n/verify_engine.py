#!/usr/bin/env python3
"""
Minimal verification script that compares ONNX (CPU) output with a TensorRT engine run
on Jetson using the name-based tensor API. All custom error handling and messaging
removed so native exceptions and messages are shown if something goes wrong.

"""
import numpy as np
import onnxruntime as ort
import tensorrt as trt
import pycuda.autoinit  # initializes CUDA context
import pycuda.driver as cuda

ONNX_PATH = "yolov8n.onnx"
ENGINE_PATH = "yolov8n_fp16.engine"
INPUT_TENSOR_NAME = "images"
OUTPUT_TENSOR_NAME = "output0"

# ONNX reference run (CPU)
sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0]
onnx_shape = tuple(int(d) if isinstance(d, int) else 1 for d in inp.shape)
onnx_in_name = inp.name

# Build random input matching ONNX input (replace dynamic dims with 1)
rng = np.random.RandomState(12345)
input_array = rng.random_sample(onnx_shape).astype(np.float32)

onnx_out = sess.run(None, {onnx_in_name: input_array})[0]

# TRT engine run (Jetson name-based API)
logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)
with open(ENGINE_PATH, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())

ctx = engine.create_execution_context()

# Query shapes/dtypes by tensor name (name-based API)
in_shape = tuple(ctx.get_tensor_shape(INPUT_TENSOR_NAME))
out_shape = tuple(ctx.get_tensor_shape(OUTPUT_TENSOR_NAME))
in_dtype = trt.nptype(engine.get_tensor_dtype(INPUT_TENSOR_NAME))
out_dtype = trt.nptype(engine.get_tensor_dtype(OUTPUT_TENSOR_NAME))

print("TRT engine input shape, dtype:", in_shape, in_dtype)
print("TRT engine output shape, dtype:", out_shape, out_dtype)

# Prepare host pinned buffers
in_elems = int(np.prod(in_shape))
out_elems = int(np.prod(out_shape))
host_in = cuda.pagelocked_empty(in_elems, in_dtype)
host_out = cuda.pagelocked_empty(out_elems, out_dtype)

# Copy input into host_in (cast to expected dtype)
src = input_array.ravel().astype(in_dtype, copy=False)
np.copyto(np.frombuffer(host_in, dtype=in_dtype, count=in_elems), src)

# Device allocations
d_in = cuda.mem_alloc(host_in.nbytes)
d_out = cuda.mem_alloc(host_out.nbytes)

# Bind by name and execute
ctx.set_tensor_address(INPUT_TENSOR_NAME, int(d_in))
ctx.set_tensor_address(OUTPUT_TENSOR_NAME, int(d_out))

stream = cuda.Stream()
cuda.memcpy_htod_async(d_in, host_in, stream)
ctx.execute_async_v3(stream_handle=stream.handle)
cuda.memcpy_dtoh_async(host_out, d_out, stream)
stream.synchronize()

trt_out = np.array(host_out).reshape(out_shape)
if trt_out.dtype == np.float16:
    trt_out = trt_out.astype(np.float32)

# Compare and print discrepancies (no tolerance checking)
onnx_v = onnx_out.ravel().astype(np.float32)
trt_v = trt_out.ravel().astype(np.float32)
n = min(onnx_v.size, trt_v.size)
diff = trt_v[:n] - onnx_v[:n]

print("ONNX output shape:", onnx_out.shape)
print("TRT  output shape:", trt_out.shape)
print("L2 diff:", np.linalg.norm(diff))
print("Max abs diff:", float(np.max(np.abs(diff))))
print("Mean abs diff:", float(np.mean(np.abs(diff))))
print("Cosine similarity:", float(np.dot(trt_v[:n], onnx_v[:n]) / (np.linalg.norm(trt_v[:n]) * np.linalg.norm(onnx_v[:n]) + 1e-12)))

# Print top differences
idx = np.argsort(-np.abs(diff))[:10]
print("\nTop differences (index, onnx, trt, diff):")
for i in idx:
    print(i, float(onnx_v[i]), float(trt_v[i]), float(diff[i]))

# Save raw flattened outputs for offline inspection
np.save("verify_onnx_output.npy", onnx_v)
np.save("verify_trt_output.npy", trt_v)
print("\nSaved verify_onnx_output.npy and verify_trt_output.npy")
