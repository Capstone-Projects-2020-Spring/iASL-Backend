import numpy as np
import requests
import sys
import json

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

address = 'http://localhost:5000'
test_url = address + '/predict'
npy_file = sys.argv[1]
frames = np.fromfile(npy_file, np.uint8).reshape(40, 150, 150, 3)
frames = np.array([preprocess(frame) for frame in frames])
response = requests.post(test_url, data=frames.tostring())
print(response.text)

scores = eval(response.text)['scores']
print(MAP[scores.index(max(scores))], max(scores))
