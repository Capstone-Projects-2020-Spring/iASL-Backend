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
WIDTH = 150
HEIGHT = 150
NUM_CHANNELS = 3

IM_WIDTH = 200
IM_HEIGHT = 200

# define the word map
#
MAP = {0:'yes',
    1:'again',
    2:'boy',
    3:'girl',
    4:'no',
    5:'ok',
    6:'help',
    7:'hello',
    8:'finish',
    9:'me'}

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
    inp = inp[:,:,::-1].astype(np.float32)
    inp /= 127.5
    inp -= 1
    return inp

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
    np_vid = np_buff.reshape(1, NUM_FRAMES, HEIGHT, WIDTH, NUM_CHANNELS)
    np_vid = np.array([preprocess(frame) for frame in np_vid])

    # get the output of the model
    #
    scores = model.predict(np_vid)[0].tolist()

    # construct a response
    #
    response = jsonpickle.encode({'scores':scores})
    
    # return a Response
    #
    return Response(response=response, status=200, mimetype="application/json")
#
# end of function


# route the http post to this method
#
@app.route('/predict_img', methods=['POST'])
def predict_img():

    # get the request
    #
    req = request
    np_buff = np.frombuffer(base64.b64decode(req.form.get('img')), np.uint8)

    # convert string data to np array
    #
    np_img = np_buff.reshape(IM_HEIGHT, IM_WIDTH, NUM_CHANNELS)[:,:,::-1]
    cv2.imsave('image.jpg', np_img)
    return Response(response="", status=200, mimetype="application/json")

# start flask app
#
app.run(host="0.0.0.0", port=8080)
