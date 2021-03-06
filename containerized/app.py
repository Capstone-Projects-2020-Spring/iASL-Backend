#!/usr/bin/env python
#
# file: $iASL_BASE/containerized/train.py
#
# revision history:
#  20200403 (TE): first version
#
# usage:
#  python app.py [mdl_dir]
#
# This script acts as a server running inside a docker
# container that handles prediction requests
#------------------------------------------------------------------------------

# import system modules
#
import os
import sys
import random
import base64

# import keras/tf modules
#
import tensorflow as tf
from tensorflow.keras.models import model_from_json

# import image processing tools
#
import numpy as np

# import flask modules
#
from flask import Flask, request, Response
import jsonpickle
import cv2

#-----------------------------------------------------------------------------
#
# global variables are listed here
#
#-----------------------------------------------------------------------------

# define model vars
#
MDL_DIR = os.path.join(os.getcwd(), "mdl_dir")
MDL_JSON = "vid_model_architecture.json"
MDL_WGT = "vid_weights-epoch.hdf5"
NUM_FRAMES = 40
WIDTH = 50
HEIGHT = 50
NUM_CHANNELS = 3

# define the word map
#
MAP = {0:'time',
       1:'think',
       2:'boy',
       3:'in',
       4:'link',
       5:'angry',
       6:'friend',
       7:'then',
       8:'language',
       9:'girl',
       10:'seperate',
       11:'food',
       12:'learn',
       13:'familiar',
       14:'for',
       15:'late',
       16:'sometimes',
       17:'ask',
       18:'yes',
       19:'and',
       20:'later',
       21:'door',
       22:'give',
       23:'look',
       24:'list',
       25:'nothing',
       26:'fingerspell',
       27:'usa',
       28:'why',
       29:'which',
       30:'tell',
       31:'come',
       32:'move',
       33:'but',
       34:'weekend',
       35:'use',
       36:'far',
       37:'me',
       38:'how',
       39:'get',
       40:'because',
       41:'stay',
       42:'good',
       43:'some',
       44:'ok',
       45:'no',
       46:'hello',
       47:'finish',
       48:'out',
       49:'also',
       50:'lab',
       51:'again',
       52:'help',
       53:'they',
       54:'del'}

#-----------------------------------------------------------------------------
#
# the main program is here
#
#-----------------------------------------------------------------------------

# load the model
#
model_f_cont = open(os.path.join(MDL_DIR, MDL_JSON), "r")
model = model_f_cont.read()
model = model_from_json(model)
model_f_cont.close()
model.load_weights(os.path.join(MDL_DIR, MDL_WGT))

# define the application
#
app = Flask(__name__)

def preprocess(inp):
    inp = cv2.resize(inp, (150, 150), interpolation = cv2.INTER_AREA)
    inp = inp.astype(np.float32)
    inp /= 127.5
    inp -= 1
    return inp
#
# end of function


# route the http post to this method
#
@app.route('/predict', methods=['POST'])
def predict():

    # get the request
    #
    req = request
    np_buff = np.frombuffer(base64.b64decode(req.form.get('vid_stuff')), np.uint8)
    
    # convert string data to np array
    #
    np_vid = np_buff.reshape(NUM_FRAMES, WIDTH, HEIGHT, NUM_CHANNELS)
    np_vid = np.expand_dims(np.array([preprocess(frame) for frame in np_vid]), axis=0)

    # get the output of the model
    #
    scores = {MAP[i]:score for i,score in enumerate(model.predict(np_vid)[0].tolist())}

    # construct a response
    #
    response = jsonpickle.encode({'scores':scores})    
    
    # return a Response
    #
    return Response(response=response, status=200, mimetype="application/json")
#
# end of function


# start flask app
#
app.run(host="0.0.0.0", port=8080, threaded=True)
