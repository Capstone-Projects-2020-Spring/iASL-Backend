#!/usr/bin/env python
#
# file: $iASL_SCRIPTS/train.py
#
# revision history:
#  20200219 (TE): first version
#
# usage:
#  import as a module
#
# This script contains data collection tools for images
# and files
#------------------------------------------------------------------------------

# import system modules
#
import os
import sys
import random
from math import ceil

# import third party modules
#
import numpy as np
from PIL import Image, ExifTags
from keras.utils import Sequence, to_categorical


#-----------------------------------------------------------------------------
#
# helper functions are listed here
#
#-----------------------------------------------------------------------------


# function: get_lines
#
# arguments: fname - path to file
#
# return: tokenized - each line of the file
#
# This function is used to parse a file list or ref file
#
def get_lines(fname):

    # try opening the file...
    #
    try:
        fp = open(fname, "r")
    except IOError as e:
        print("[%s]: %s" % (sys.argv[0], e))
        exit(-1)

    # split by whitespace
    #
    tokenized = fp.read().split()

    # close the file
    #
    fp.close()

    # exit gracefully
    #
    return tokenized
#
# end of function


# function: get_grayscale_img(ifile)
#
# arguments: ifile - the path to the file
#
# return: np_img - array representation of the image
#
# This is the main function
#
def get_grayscale_img(ifile):

    # try to open the file using PIL
    #
    try:
        img_obj = Image.open(ifile)
    except IOError as e:
        print("[%s]: %s" % (sys,argv[0], e))
        exit(-1)

    # if we have EXIF data
    #
    if(img_obj._getexif() is not None):

        # get all the exif tags
        #
        exif=dict((ExifTags.TAGS[k], v) for k, v in img_obj._getexif().items() if k in ExifTags.TAGS)

        # rotate if it has been auto rotated
        #
        if   exif['Orientation'] == 3 : 
            img_obj=img_obj.rotate(180, expand=True)
        elif exif['Orientation'] == 6 : 
            img_obj=img_obj.rotate(270, expand=True)
        elif exif['Orientation'] == 8 : 
            img_obj=img_obj.rotate(90, expand=True)

    # convert the image to grayscale
    #
    img_obj = img_obj.convert('L').resize((200, 200))

    # load the image
    #
    img_obj.load()

    # convert to a numpy array and reshape
    #
    np_img = np.asarray(img_obj, dtype="int32").reshape(200, 200, 1)

    # exit gracefully
    #
    return np_img
#
# end of function


#-----------------------------------------------------------------------------
#
# the data generator class is listed here
#
#-----------------------------------------------------------------------------

# define the DataGenerator class
#
class DataGenerator(Sequence):

    # function: init
    #
    # arguments: flist - list of image file paths
    #            labels - corresponding labels to the file
    #            batch_size - number of images to load in a batch
    #            class_mapping - dictionary mapping of labels and ids
    #
    # return: none
    #
    # This is the constructor for the class
    #
   def __init__(self, flist, labels, batch_size, class_mapping):

       # initialize the object
       #
       self.files = flist
       self.lbl_mapping = class_mapping
       self.labels = labels
       self.batch_size = batch_size
       self.num_classes = len(self.lbl_mapping)
       self.init_mapping()
    #
    # end of function

    
    # function: __len__
    #
    # arguments: none
    #
    # return: int - number of batches
    #
    # This method returns the number of batches in the dataset
    #
    def __len__(self):

        # calculate as ceil(N/batch_size)
        #
        return int(ceil(float(len(self.labels)) / self.batch_size))
    #
    # end of function
    
    
    # function: init_mapping
    #
    # arguments: none
    #
    # return: none
    #
    # This method generates a master mapping of images and their labels
    # [[img_1, lbl_1], [img_2, lbl_2]...]
    #
    def init_mapping(self):

        # if there is a mismatch
        #
        if(len(self.files) != len(self.labels)):
            print("[%s]: number of files (%d) does not match number of labels (%d)" \
                  % (sys.argv[0], len(self.files), len(self.labels)))
            exit(-1)

        # initialize mapping
        #
        self.mapping = []

        # for each file and label
        #
        for fname, lbl in zip(self.files, self.labels):

            # add the pair as a list to the map
            #
            self.mapping.append([fname, lbl])

        # shuffle the dataset
        #
        random.shuffle(self.mapping)
    #
    # end of function
    
    
    # function: __getitem__
    #
    # arguments: idx - the id of the batch
    #
    # return: X, y - inputs and target labels for the batch
    #
    # This method returns a batch of k images and labels, where
    # k is the batch_size
    #
    def __getitem__(self, idx):

        # slice the mapping properly
        #
        mapping_slice = self.mapping[idx*self.batch_size:(idx+1) * self.batch_size]

        # initialize the inputs and labels
        #
        inputs = []
        labels = []

        # for each [image, label] pair in the slice
        #
        for entry in mapping_slice:

            # unpack the entry
            #
            ifile, lbl = entry

            # get the integer representation of the label
            #
            label = self.lbl_mapping[lbl]

            # read the image and append it to the inputs
            #
            inputs.append(get_grayscale_img(ifile).copy())

            # add the id label to list of labels
            #
            labels.append(label)

        # return np array of images and np one-hot-encoded vector
        # of the labels
        #
        return np.asarray(inputs), to_categorical(labels, num_classes=self.num_classes)
    #
    # end of function
#
# end of class

#
# end of file
