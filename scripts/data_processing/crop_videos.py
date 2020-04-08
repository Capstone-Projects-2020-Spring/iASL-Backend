# This script automates running many batch commands to trim ASL word clips
# out of raw video obtained with batch_download.py.
# To work properly, FFMPEG must be installed.

#a subdirectory called "words" must be created in the folder this script is run from
#the videos to be cropped must be contained in a subdirectory called "raw"

# Imports
import json
import os
import sys

# Function to run batch command using FFMPEG to trim video based on start, end parameters
def crop(start, end, infile, outfile):
    str = "ffmpeg -i raw/" + infile + " -ss " + start + " -to " + end + " -c copy \"words/" + outfile + "\""
    print(str)
    os.system(str)
    
# Open Microsoft JSON to data set
with open(sys.argv[1], 'r') as f:
    dataset=json.load(f)

# Get the video IDs that are used as filenames in the other directory
# name_occurences is used to track the number of clips of the same words 
name_occurences = {}
ct_loop = 0

# vid_lengths = [] commented out code is for getting average clip length

for x in dataset[0:100]: # REMOVE [BOUNDS] FOR WHOLE DATA SET   
    start = x['start_time']
    end = x['end_time']
    if(end-start < 2.0):
        start = end - 2.5
    
    # vid_lengths.append(end-start)
    
    match_url = x['url'].split('=', 1)[1].split('&', 1)[0]
    name = x['clean_text']

    # Increment the occurence counter if the name is in name_occurences
    if(name in name_occurences):
        name_occurences[name] += 1
    else:
        name_occurences[name] = 0

    filename = name + '_' + str(name_occurences[name])
    crop(str(start), str(end), match_url + ".mp4", filename + ".mp4")

# print("avg vid length:" + str(sum(vid_lengths)/len(vid_lengths)))
