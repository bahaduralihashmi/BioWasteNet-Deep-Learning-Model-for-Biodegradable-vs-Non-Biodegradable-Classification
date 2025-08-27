from onnx_tf.backend import prepare
import onnx

# Load the ONNX model
onnx_model = onnx.load(r"C:\Users\bahad\OneDrive\Desktop\ultralytics\runs\detect\Person_detect\weights\best.onnx")

# Convert ONNX to TensorFlow
tf_rep = prepare(onnx_model)

# Export TensorFlow SavedModel
tf_rep.export_graph("Bio_non_model")
