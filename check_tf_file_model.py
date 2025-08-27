import tensorflow as tf
import cv2
import numpy as np

# Load exported SavedModel
model = tf.saved_model.load("Bio_non_model")

# Test image
img = cv2.imread(r"C:\Users\bahad\OneDrive\Desktop\ultralytics\images\Image1.jpg", cv2.IMREAD_GRAYSCALE)
img_resized = cv2.resize(img, (96, 96)) / 255.0
img_tensor = np.expand_dims(img_resized, axis=(0,1)).astype(np.float32)  # (1,1,96,96)

# Run inference
infer = model.signatures["serving_default"]
output = infer(tf.constant(img_tensor))

print("Output keys:", output.keys())
for k,v in output.items():
    print(k, v.shape)
    print(v.numpy()[:5])
