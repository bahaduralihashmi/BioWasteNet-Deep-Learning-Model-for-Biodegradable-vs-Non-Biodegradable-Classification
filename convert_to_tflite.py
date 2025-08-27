import tensorflow as tf
import numpy as np

saved_model_dir = r"C:\Users\bahad\OneDrive\Desktop\ultralytics\tf_model"

# Representative dataset for INT8 calibration
def representative_data_gen():
    for _ in range(100):
        # Match the model input: (1, 1, 64, 64), dtype float32
        dummy = np.random.rand(1, 1, 64, 64).astype(np.float32)
        yield [dummy]

# Create TFLite converter
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

# Enable optimizations
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

# Force full INT8 quantization (ESP32-friendly)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Convert the model
tflite_model = converter.convert()

# Save the quantized model
with open("model_int8.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ INT8 TFLite model saved as model_int8.tflite")
