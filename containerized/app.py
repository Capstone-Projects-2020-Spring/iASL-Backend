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

# route the http post to this method
#
@app.route('/predict', methods=['POST'])
def predict():

    # get the request
    #
    req = request

    # convert string data to np array
    #
    np_vid = np.fromstring(req.data, np.float32)
    np_vid = np_vid.reshape(1, NUM_FRAMES, HEIGHT, WIDTH, NUM_CHANNELS)

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


# start flask app
#
app.run(host="0.0.0.0", port=8080)
