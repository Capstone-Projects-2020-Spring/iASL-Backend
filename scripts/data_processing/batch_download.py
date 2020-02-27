#!/usr/bin/env python
#
# file: $iASL_SCRIPTS/data_processing/batch_download.py
#
# revision history:
#  20200225 (TE): refactored code to seperate different labels
#  20200219 (AL): first version
#
# usage:
#  python batch_download.py [JSON] [raw_odir] [clean_odir]
#
# This script downloads youtube videos from the MS-ASL json file and
# places them in their own directory
#------------------------------------------------------------------------------

# import system modules
#
from pytube import YouTube, Stream
import json
import sys
import os

#-----------------------------------------------------------------------------
#
# global variables are listed here
#
#-----------------------------------------------------------------------------

# define general global variables
#
NUM_ARGS = 3
MP4_EXT = ".mp4"

#-----------------------------------------------------------------------------
#
# helper functions are listed here
#
#-----------------------------------------------------------------------------


# function: crop
#
# arguments: start - the start time to begin cropping
#            end - the end time to stop cropping
#            infile - the name of the video that was downloaded
#            outfile - the name of the output file to create
#            raw_odir - the directory where the infile was stored
#            clean_odir - the output directory path
#            label - the label of the clip
#
# return: int - exit status code
#
# This functions runs the ffmpeg command on the given video clip
# and outputs a file in label/label_x.mp4
#
def crop(start, end, infile, outfile, raw_odir, clean_odir, label):

    # the path of the downloaded video clip
    #
    in_path = os.path.join(raw_odir, infile)

    # the path of the output file
    #
    out_path = os.path.join(clean_odir, label, outfile)

    # create the output directory if it does not exist
    #
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))

    # format the command
    #
    cmd = "ffmpeg -i " + in_path + MP4_EXT + " -ss " \
          + start + " -to " + end + " -c copy " + out_path

    # execute the command
    #
    return os.system(cmd)
#
# end of function


# function: gen_vid_mapping
#
# arguments: json_fname - path to the json file
#
# return: vid_mapping - dictionary mapping
#
# This functions parses through the MS-ASL JSON file and
# creates a dictionary mapping of format:
# {url: [[s1, e1, lbl1], [s2, e2, lbl2] ...] ...}
#
def gen_vid_mapping(json_fname): 

    # open the json file and read 
    #
    with open(json_fname, 'r') as fp:
        dataset = json.load(fp)

    # instantiate a dictionary
    #
    vid_mapping = {}

    # for each json object
    #
    for obj in dataset:

        # remove the &t if it exists
        #
        cut_url = obj['url'].split('&', 1)[0]

        # get the start and stop time
        #
        start = obj['start_time']
        end = obj['end_time']

        # extend the video if the duration is too short
        #
        if(end - start < 2.0):
            start = end - 2.5

        # grab the label, replace space with underscore
        #
        label = obj['clean_text'].replace(" ", "_")

        # add the start, end, and label to the dictionary
        #
        if cut_url in vid_mapping:
            vid_mapping[cut_url].append([start, end, label])
        else:
            vid_mapping[cut_url] = [[start, end, label]]

    # exit gracefully
    #
    return vid_mapping
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

    # if the user did not provide the correct amount of args
    #
    if(len(argv) != NUM_ARGS):
        print("usage: python batch_download.py [JSON] [odir] [clean_odir]")
        exit(-1)

    # set local variables
    #
    json_train = argv[0]
    raw_odir = argv[1]
    clean_odir = argv[2]

    # if the output path does not exist, make it
    #
    if not os.path.exists(raw_odir):
        print("making directory: %s" % (raw_odir))
        os.makedirs(raw_odir)
    
    # create dict mapping of URL: [[start, end, label],[start, end, label]..]
    #
    vid_mapping = gen_vid_mapping(json_train)
    
    # display informational message
    #
    print("Number of videos:", len(vid_mapping))

    # set the number of files attempted, error
    #
    num_unavailable = 0
    num_downloaded = 0

    # create mapping of labels with key as label
    # and value as occurence
    #
    labels = {}

    # for each url in the mapping
    #
    for url in vid_mapping:

        # get the name of the output video name
        #
        name = "vid_" + str(num_downloaded + 1).zfill(4)

        # try to get video stream for download
        #
        try:
            YouTube(url=url).streams.first().download(output_path=raw_odir, filename=name)
            num_downloaded += 1

        # keep going if we meet an exception
        #
        except Exception:
            print("Video " + name + " is unavailable for download.")
            num_unavailable += 1
            continue

        # for each clip of the video
        #
        for start, end, label in vid_mapping[url]:

            # increment the occurence
            #
            labels[label] = labels.get(label, 0) + 1

            # set the output file name
            #
            fname = label + "_" + str(labels[label]).zfill(3) + MP4_EXT

            # crop and save to the clean odir
            #
            crop(str(start), str(end), name, fname, raw_odir, clean_odir, label)

        # remove the temporary file
        #
        os.system('rm ' + os.path.join(raw_odir, name) + MP4_EXT)

    # display status
    #
    print(str(num_downloaded) + " videos successfully downloaded.")
    print(str(num_unavailable) + " video(s) could not be downloaded.")

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
# end of function

#
# end of file
