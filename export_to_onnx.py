from ultralytics import YOLO
import onnx

# Step 1: Load your trained PyTorch model
model = YOLO(r"C:\Users\bahad\OneDrive\Desktop\ultralytics\runs\detect\Person_detect\weights\best.pt")

# Step 2: Export to ONNX with opset 12
onnx_path = model.export(format="onnx", opset=12, imgsz=96)

# Step 3: Force ONNX IR version = 9 for compatibility
onnx_model = onnx.load(onnx_path)
onnx_model.ir_version = 9
onnx.save(onnx_model, onnx_path)

print(f"✅ ONNX model exported and fixed at: {onnx_path}")
