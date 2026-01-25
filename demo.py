import tensorflow as tf

# Print the list of physical devices recognized by TensorFlow
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print("TensorFlow is using the GPU")
else:
    print("TensorFlow is using the CPU")
