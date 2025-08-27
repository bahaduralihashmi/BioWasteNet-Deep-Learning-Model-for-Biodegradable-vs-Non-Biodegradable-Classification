import tensorflow as tf
import numpy as np
import cv2

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=r"C:\Users\bahad\OneDrive\Desktop\ultralytics\Bio_non_model_quant.tflite")
interpreter.allocate_tensors()

# Get input & output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Preprocess image
img = cv2.imread(r"C:\Users\bahad\OneDrive\Desktop\ultralytics\images\WhatsApp Image 2025-08-24 at 16.37.13_b0818ff4.jpg", cv2.IMREAD_GRAYSCALE)
img_resized = cv2.resize(img, (96, 96)) / 255.0
img_tensor = np.expand_dims(img_resized, axis=(0,1)).astype(np.float32)  # (1,1,96,96)

# Set input tensor
interpreter.set_tensor(input_details[0]['index'], img_tensor)

# Run inference
interpreter.invoke()

# Get output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])  # shape: (1, 6, N)
print("Raw output shape:", output_data.shape)
