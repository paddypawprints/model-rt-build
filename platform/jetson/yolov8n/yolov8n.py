from ultralytics import YOLO

print("Load the YOLO11 model from Ultralytics - yolov8n.pt")
model = YOLO("yolov8n.pt")

print( "Export the model to ONNX format" )
model.export(format="onnx")  # creates 'yolo11n.onnx'

print( "Load the exported ONNX model" )
onnx_model = YOLO("yolov8n.onnx")

print("Run inference")
results = onnx_model("https://ultralytics.com/images/bus.jpg")
