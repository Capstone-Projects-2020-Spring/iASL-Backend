# Reads a model trained using TensorFlow
# Saves the model as a TensorFlow Lite model

# https://www.tensorflow.org/lite/guide/get_started

import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model(directory_of_saved_model)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
