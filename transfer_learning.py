import argparse
import numpy as np
import sys

parser = argparse.argumentParser()
parser.addArgument("-n", "--newdata", help="Specify the name of the new dataset")
parser.addArgument("-w", "--weights", help="Specify the file containing the weights and biases")
# Example: python3 transfer_learning.py -n new_dataset.bin -w weights.txt

# Check whether the user entered a dataset for training; exit if no dataset entered
if (args.newdata == None)
  print("Error: No new dataset specified")
  sys.exit()
  
# Check whether the user entered weights; exit if no weights entered
if (args.weights == None)
  print("Error: No weights specified")
  sys.exit()

# Read in weights and biases
weights[][] = args.weights
w = np.matrix(weights)

# Multiply weights and biases by W = 0.7

# Read in new data to train

# Performing training on the new data, updating the above weights and biases

# Save the updated weights and biases
