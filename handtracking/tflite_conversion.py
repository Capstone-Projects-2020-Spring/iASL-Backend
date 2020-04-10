# Reads a model trained using TensorFlow
# Saves the model as a TensorFlow Lite model

# Example usage:
# python3 tflite_conversion.py directoryWithSavedModel

# TensorFlow Lite Model saved to file named "converted_model.tflite"

# Source: https://www.tensorflow.org/lite/guide/get_started

# Imports
import argparse
import sys
import tensorflow as tf

# Setup command-line arguments
parser = argparse.argumentParser()
parser.addArgument("-d", "--directory", "Name of directory containing the saved TensorFlow model to convert")

# Check whether a directory was specified; exit if none was
if args.directory == None:
  print("Error: No directory specified")
  sys.exit()

converter = tf.lite.TFLiteConverter.from_saved_model(args.directory)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
