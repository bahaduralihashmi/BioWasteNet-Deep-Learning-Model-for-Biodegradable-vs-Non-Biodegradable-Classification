import tensorflow as tf

# Convert SavedModel to quantized TFLite
converter = tf.lite.TFLiteConverter.from_saved_model("Bio_non_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("Bio_non_model_quant.tflite", "wb") as f:
    f.write(tflite_model)

print("Quantized TFLite model saved as Bio_non_model_quant.tflite")
