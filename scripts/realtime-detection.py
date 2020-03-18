#!/usr/bin/env python
#
# file: $iASL_SCRIPTS/realtime-detection.py
#
# revision history:
#  20200219 (TE): first version
#
# usage:
#  python realtime-detection.py [params]
#
# This script begins a video capture stream and predicts each frame
# of the stream using a trained model
#------------------------------------------------------------------------------

# import system modules
#
import sys
import os
import time
import threading
import queue

# import keras/tf modules
#
import tensorflow as tf
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
from keras.models import model_from_json
from keras.applications.inception_v3 import preprocess_input

# import image tools
#
import numpy as np
import cv2

# import custom modules
#
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
NUM_ARGS = 1
NEW_LINE = "\n"
INITIAL_DELAY = float(3.0)

#-----------------------------------------------------------------------------
#
# helper functions are listed here
#
#-----------------------------------------------------------------------------


# function: worker
#
# arguments: frame_q - the frame queue
#            score_q - the scores queue
#            model - the model used for inference
#
# return: none
#
# This is the function for a worker thread to process
# an image
#
def worker(frame_q, score_q, model):

    # run indefinitely
    #
    while True:

        # get a frame from the queue
        #
        try:
            frame = frame_q.get()
        except queue.Empty:
            continue

        # run through the model
        #
        scores = model.predict(frame.reshape(1, 200, 200, 3))

        # put the scores in the scores queue
        #
        score_q.put(scores)
#
# end of function


# function: show_statistics
#
# arguments: letters - list of letters to display
#
# return: none
#
# This function displays a black screen letters predicted
#
def show_statistics(letters):

    # create a black matrix
    #
    text_image = np.zeros((300, 512, 3), np.uint8)

    # put the class_name
    #
    cv2.putText(text_image, "".join(letters),
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)

    # display the result
    #
    cv2.imshow("Result", text_image)
#
# end of function


# function: add_letter
#
# arguments: letters - list of letters to display
#            letter - new letter to add
#
# return: letters - updated list of letters
#
# This is a wrapper function that adds letters to the list
#
def add_letter(letters, letter):

    # if this is nothing return original list
    #
    if letter == 'nothing':
        return letters

    # if this is delete, remove last element
    #
    elif letter == 'del':

        # check if there is any letters to remove
        #
        if len(letters) > 0:
            letters.pop()
            return letters
        return letters

    # add a space if this is space
    #
    elif letter == 'space':
        letters.append(' ')
        return letters

    # this is a normal letter, append and return
    #
    else:
        letters.append(letter)
        return letters
#
# end of function


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

    # if not enough arguments were provided
    #
    if len(argv) != NUM_ARGS:
        print("usage: python decode.py [params]")
        exit(-1)

    # parse through the parameter file
    #
    param_obj = ParamParser(argv[0])

    # convert the keys of the mapping to integers
    #
    mapping = {int(key):value for key,value in param_obj["MAP"].items()}

    # load the "decode values" section of param file
    #
    decode_values = param_obj["DECODE_VALUES"]

    # grab the model directory
    #
    mdl_dir = iASL_OUT + decode_values["mdl_dir"]

    # open the json acrhitecture
    #
    model_f_cont = open(os.path.join(mdl_dir, decode_values["mdl_name"]) , "r")

    # read the contents
    #
    model = model_f_cont.read()

    # close the file pointer
    #
    model_f_cont.close()

    # create a model from the JSON file
    #
    loaded_model = model_from_json(model)

    # load the weights from the given file
    #
    loaded_model.load_weights(os.path.join(mdl_dir, decode_values["wgt_name"]))
    
    # instantiate an object tracker
    #
    tracker = cv2.TrackerMIL_create()

    # begin video capture
    #
    video = cv2.VideoCapture(0)

    # if we failed to open the camera
    #
    if not video.isOpened():
        print("[%s]: failed to open the camera" % (sys.argv[0]))
        sys.exit(-1)

    # define the initial bounding box
    #
    top, right, bottom, left = 20, 20, 220, 220

    # initially, we did not instantiate the tracker
    #
    not_set = True

    # get the current time
    #
    s1 = time.time()
    index = 0

    # instantiate thread safe queues
    #
    frame_q = queue.Queue()
    score_q = queue.Queue()

    # launch a new worker thread
    #
    threading.Thread(target=worker, args=(frame_q, score_q, loaded_model)).start()

    # keep track of the last k predictions
    #
    top_k_preds = []

    # keep track of the letters so far
    #
    letters = []

    # keep track if we are adding the same letter
    #
    same_letter = 0

    # check if this is a delete
    #
    is_del = False

    # number of consecutive del predictions
    #
    num_del = 0

    # loop indefinitely
    #
    while(True):

        # read the frame from the video
        #
        grabbed, frame = video.read()

        # flip the image
        #
        frame = cv2.flip(frame, 1)

        # if we did not set the tracker and
        # 3 seconds has elapsed
        #
        if not_set and time.time() - s1 > INITIAL_DELAY:

            # initialize the tracker
            #
            tracker.init(frame, (20, 20, 200, 200))

            # tracker is now set
            not_set = False
            continue

        # if we did not set the tracker but it
        # is less than 3 seconds
        #
        elif not_set:

            # draw a rectange over the initial bounding box
            #
            cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)

            # display the frame
            #
            cv2.imshow("Video feed", frame.astype('uint8'))

            # exit if esc key is pressed
            #
            k = cv2.waitKey(1) & 0xff
            if k == 27 : break
            continue

        # get the bbox of the updated hand object
        #
        _, bbox = tracker.update(frame)

        # define the left, top, right, bottom regions of the bbox
        #
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

        # draw a blue rectangle over the detected hand
        #
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)

        # set the dimensions
        #
        left, top = p1
        right, bottom = p2

        # grab the region of interest (hand)
        #
        roi = np.asarray(frame[top:bottom, left:right], dtype='uint8').copy()

        # display the frame
        #
        cv2.imshow("Video feed", frame)

        # exit if esc key is pressed
        #
        keypress = cv2.waitKey(1) & 0xFF
        if k == 27: break

        roi = cv2.resize(roi, (200, 200), interpolation = cv2.INTER_AREA)[:, :, ::-1]
        roi = preprocess_input(roi)
        frame_q.put(roi)
        try:
            scores = score_q.get()
            max_ind = np.argmax(scores)
            max_score = np.max(scores)
            if len(top_k_preds) < K:
                top_k_preds.append((max_ind, max_score))
            elif len(top_k_preds) == K:
                top_k_preds.pop(0)
                top_k_preds.append((max_ind, max_score))
            if len(set([x[0] for x in top_k_preds])) != 1:
                continue
            if float(sum([x[1] for x in top_k_preds])) / float(K) < THRESH:
                continue
            if mapping[max_ind] == 'del':
                is_del = True
                num_del += 1
            else:
                is_del = False
                num_del = 0
            if(len(letters) == 0):
                letters = add_letter(letters, mapping[max_ind])
            elif(mapping[max_ind] == letters[-1] or (mapping[max_ind] == 'space' and letters[-1] == ' ' ) or (num_del > 1)):
                same_letter += 1
                if(same_letter == 15 or (is_del and same_letter == 5)):
                    letters = add_letter(letters, mapping[max_ind])
                    same_letter = 0                    
            else:
                same_letter = 0
                letters = add_letter(letters, mapping[max_ind])
        except queue.Empty:
            continue            
        show_statistics(letters)            

    # exit gracefully
    #
    return True
#
# end of function


# begin gracefully
#
if __name__ == '__main__':
    main(sys.argv[1:])
#
# end of file
