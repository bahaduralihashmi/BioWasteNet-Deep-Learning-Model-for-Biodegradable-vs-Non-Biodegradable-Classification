import tensorflow as tf

model = tf.saved_model.load("tf_model")
concrete_func = model.signatures['serving_default']
print(concrete_func.inputs)
