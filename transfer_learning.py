import argparse
import sys

parser = argparse.argumentParser()
parser.addArgument("-n", "--newdata", help="Specify the name of the new dataset")
# Example: python3 transfer_learning.py -n new_dataset

# Check whether the user entered a dataset; exit if no dataset entered
if (args.newdata == None)
  print("Error: No new dataset specified")
  sys.exit()

# Read in weights and biases

# Multiply weights and biases by W = 0.7

# Read in new data to train

# Performing training on the new data, updating the above weights and biases

# Save the updated weights and biases
