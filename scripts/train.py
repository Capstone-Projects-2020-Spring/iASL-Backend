#!/usr/bin/env python
#
# file: $iASL_SCRIPTS/train.py
#
# revision history:
#  20200219 (TE): first version
#
# usage:
#  python train.py [params] [odir]
#
# This script trains a DL model on an ASL dataset
#------------------------------------------------------------------------------

# import system modules
#
import os
import sys
import random

# import keras/tf modules
#
from keras import backend as K
from keras.models import Sequential
from keras.layers import Input, GlobalAveragePooling3D
from keras.layers import Dense, Flatten, Dropout
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from i3d import *

# import custom modules
#
import data_processing.collect_data as cd
from params_tool import ParamParser

#-----------------------------------------------------------------------------
#
# global variables are listed here
#
#-----------------------------------------------------------------------------

# define environment variables
#
iASL_BASE = os.environ['iASL_BASE']
iASL_SCRIPTS = os.environ['iASL_SCRIPTS']
iASL_OUT = os.environ['iASL_OUT']
iASL_DATA = os.environ['iASL_DATA']
iASL_LISTS = os.environ['iASL_LISTS']
iASL_PARAMS = os.environ['iASL_PARAMS']

# define general global variables
#
NUM_ARGS = 2
TEMP_WEIGHT_FILE = "weights-epoch-{epoch:d}.hdf5"
MAP = "MAP"

#-----------------------------------------------------------------------------
#
# the main program is here
#
#-----------------------------------------------------------------------------


# function: main
#
# arguments: argv - commandline arguments
#
# return: boolean - logical status value
#
# This is the main function
#
def main(argv):

    # if the user did not provide the correct amount of args
    #
    if(len(argv) != NUM_ARGS):
        print("usage: python train.py [params] [odir]")
        exit(-1)

    # set the output directory
    #
    odir = argv[1]

    # try to make the directory if it does not exist
    #
    if not os.path.exists(odir):
        try:
            os.makedirs(odir)
        except OSError as e:
            print("[%s]: %s" % (sys.argv[0], e))
            exit(-1)

    # create a param parser object
    #
    param_obj = ParamParser(argv[0])

    # get the class mapping
    #
    class_mapping = param_obj["MAP"]

    # get the training values section of param
    #
    train_values = param_obj["TRAIN_VALUES"]

    # get the file list and annotation file
    #
    flist = iASL_LISTS + train_values["train_list"]
    anno = iASL_LISTS + train_values["train_labels"]

    # get the actual files and labels
    #
    files = cd.get_flist(flist)
    labels = cd.get_lines(anno)

    # randomly split the dataset
    #
    temp_zipped = list(zip(files, labels))
    random.shuffle(temp_zipped)
    files, labels = zip(*temp_zipped)

    # get the split between train/cv
    #
    split = float(train_values["tr_cv_split"])

    # set the batch size
    #
    batch_size = int(train_values["batch_size"])

    # set the number of epochs
    #
    epochs = int(train_values["num_epochs"])

    # set the weight file
    #
    tmp_weights = os.path.join(odir, TEMP_WEIGHT_FILE)

    # save weights each epoch
    #
    checkpoint = ModelCheckpoint(tmp_weights)

    # create a split between test/cross validation
    #
    X_train = files[:int(split*len(files))]
    X_test = labels[:int(split*len(labels))]
    Y_train = files[int(split*len(files)):]
    Y_test = labels[int(split*len(labels)):]

    # create two generator objects (one for train, one for cv)
    #
    tr_dat_obj = cd.DataGenerator(X_train, X_test, batch_size, class_mapping, eval(train_values['is_vid']))
    cv_dat_obj = cd.DataGenerator(Y_train, Y_test, batch_size, class_mapping, eval(train_values['is_vid']))

    # define the model architecture (this changes a lot...)
    #
    inp = Input(shape=(40, 200, 200, 3))
    base_model = Inception_Inflated3d(include_top=False, weights='rgb_imagenet_and_kinetics', input_tensor=inp, classes=tr_dat_obj.num_classes)
    x = base_model.output
    x = GlobalAveragePooling3D()(x)
    x = Dense(1024, activation='relu')(x)
    out = Dense(tr_dat_obj.num_classes, activation='softmax')(x)
    for layer in base_model.layers:
        layer.trainable = False
        if isinstance(layer, keras.layers.normalization.BatchNormalization):
            layer.trainable = True

    # connect the input and output layers
    #
    model = Model(inputs=base_model.input, outputs=out)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    # fit the train/cv generators to the model
    #
    model.fit_generator(generator=tr_dat_obj, epochs=epochs, verbose=1, max_queue_size=10, \
                        workers=6, shuffle=False, callbacks=[checkpoint], validation_data=cv_dat_obj)

    # conver the model to a json
    #
    model_json = model.to_json()

    # save the model as a json file
    #
    with open(os.path.join(odir, train_values['mdl_name']), "w") as json_arch:
        json_arch.write(model_json)

    # save the final model weights
    #
    model.save(os.path.join(odir, train_values['wgt_name']), overwrite=True)

    # exit gracefully
    #
    return True
#
# end of main


# begin gracefully
#
if __name__ == '__main__':
    main(sys.argv[1:])
#
# end of file
