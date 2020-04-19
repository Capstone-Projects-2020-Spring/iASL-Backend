import numpy as np
import requests
import sys
import json
import base64
import cv2

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

def preprocess(inp):
    inp = inp[:,:,::-1].astype(np.float32)
    inp /= 127.5
    inp -= 1
    return inp

address = 'http://localhost:8080'
test_url = address + '/predict_img'
npy_file = sys.argv[1]
frames = np.fromfile(npy_file, np.uint8).reshape(40, 150, 150, 3)
img = cv2.resize(frames[0], (200, 200), cv2.INTER_AREA)
response = requests.post(test_url, data={'img':base64.b64encode(img)})
print(response.text)
