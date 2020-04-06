# iASL Transfer Learning
# Read in a dataset to be trained and weights generated by training another dataset
# Reduce the "weight" given to the previous weights and train the new dataset, updating the previous weights

import argparse
import numpy as np
import os
import sys

# Experimental code for multithreading
config = tf.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS, inter_op_parallelism_threads=4, allow_soft_placement=True, device_count = {'CPU': NUM_PARALLEL_EXEC_UNITS})
os.environ["OMP_NUM_THREADS"] = "NUM_PARALLEL_EXEC_UNITS"
os.environ["KMP_BLOCKTIME"] = "1"
os.environ["KMP_SETTINGS"] = "TRUE"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"

# Experimental code for GPU support
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# Setup command-line arguments
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
for i in range(np.size(w, 0)):
  for j in range(np.size(w, 1)):
    w[i][j] *= 0.7

# Read in new data to train
newdata[] = args.newdata

'''Perform training on the new data, updating the above weights and biases'''
# Note: Using some code from scripts/train.py (by @TarekE-dev)
epochs = 5
batch_size = 16

files = newdata.get_flist(flist)
labels = newdata.get_lines(anno)

# Randomly split the dataset
temp_zipped = list(zip(files, labels))
random.shuffle(temp_zipped)
files, labels = zip(*temp_zipped)
split = float(train_values["tr_cv_split"])

tmp_weights = os.path.join(odir, TEMP_WEIGHT_FILE)
checkpoint = ModelCheckpoint(tmp_weights)

# Split the files
X_train = files[:int(split * len(files))]
X_test = labels[:int(split * len(labels))]
Y_train = files[int(split * len(files)):]
Y_test = labels[int(split * len(labels)):]

# Define the model (same as in scripts/train.py)
model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), strides = (1,1), activation='relu', \
                 input_shape=(200, 200, 1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size = (3, 3), strides = (2,2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size = (3, 3), strides = (2,2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(tr_dat_obj.num_classes, activation = 'softmax'))

adam_opt = Adam(lr=float(train_values["learning_rate"]))  # Optimizer (set to optimize the learning rate)

model.compile(loss='categorical_crossentropy', optimizer= adam_opt, \
                  metrics=['accuracy'])

model.fit_generator(generator=tr_dat_obj, epochs=epochs, verbose=1, max_queue_size=10, \
                        workers=6, shuffle=False, callbacks=[checkpoint], validation_data=cv_dat_obj)

model_json = model.to_json()

# Save the model
with open(os.path.join(odir, train_values['mdl_name']), "w") as json_arch:
        json_arch.write(model_json)

model.save(os.path.join(odir, train_values['wgt_name']), overwrite=True)
