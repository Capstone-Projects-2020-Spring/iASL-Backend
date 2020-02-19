import os
import sys
import random
import numpy as np
from PIL import Image, ExifTags
from keras.utils import Sequence, to_categorical
from math import ceil

def get_lines(fname):
    try:
        fp = open(fname, "r")
    except IOError as e:
        print("[%s]: %s" % (sys.argv[0], e))
        exit(-1)
    files = fp.read().split()
    fp.close()
    return files

def get_grayscale_img(ifile):
    try:
        img_obj = Image.open(ifile)
    except IOError as e:
        print("[%s]: %s" % (sys,argv[0], e))
        exit(-1)
    if(img_obj._getexif() is not None):
        exif=dict((ExifTags.TAGS[k], v) for k, v in img_obj._getexif().items() if k in ExifTags.TAGS)
        if   exif['Orientation'] == 3 : 
            img_obj=img_obj.rotate(180, expand=True)
        elif exif['Orientation'] == 6 : 
            img_obj=img_obj.rotate(270, expand=True)
        elif exif['Orientation'] == 8 : 
            img_obj=img_obj.rotate(90, expand=True)
    img_obj = img_obj.convert('L').resize((200, 200))
    img_obj.load()
    np_img = np.asarray(img_obj, dtype="int32").reshape(200, 200, 1)
    return np_img

class DataGenerator(Sequence):
    def __init__(self, flist, labels, batch_size, class_mapping):
        self.files = flist
        self.lbl_mapping = class_mapping
        self.labels = labels
        self.batch_size = batch_size
        self.num_classes = len(self.lbl_mapping)
        self.init_mapping()

    def __len__(self):
        return int(ceil(float(len(self.labels)) / self.batch_size))

    def init_mapping(self):
        if(len(self.files) != len(self.labels)):
            print("[%s]: number of files (%d) does not match number of labels (%d)" \
                  % (sys.argv[0], len(self.files), len(self.labels)))
            exit(-1)
        temp_zipped = list(zip(self.files, self.labels))
        random.shuffle(temp_zipped)
        self.files, self.labels = zip(*temp_zipped)
        self.mapping = []
        for fname, lbl in zip(self.files, self.labels):
            self.mapping.append([fname, lbl])
        random.shuffle(self.mapping)
        return
    
    def __getitem__(self, idx):
        mapping_slice = self.mapping[idx*self.batch_size:(idx+1) * self.batch_size]
        inputs = []
        labels = []
        for entry in mapping_slice:
            ifile, lbl = entry
            label = self.lbl_mapping[lbl]
            inputs.append(get_grayscale_img(ifile).copy())
            labels.append(label)
        return np.asarray(inputs), to_categorical(labels, num_classes=self.num_classes)
