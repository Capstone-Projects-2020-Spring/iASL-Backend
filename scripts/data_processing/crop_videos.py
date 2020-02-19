import json
import os

#a subdirectory called "words" must be created in the folder this script is run from
#the videos to be cropped must be contained in a subdirectory called "raw"

def crop(start, end, infile, outfile):
    str = "ffmpeg -i raw/" + infile + " -ss " + start + " -to " + end + " -c copy \"words/" + outfile + "\""
    print(str)
    os.system(str)
    
with open('MSASL_train.json', 'r') as f:
    dataset=json.load(f)

name_occurences = {}
ct_loop = 0

for x in dataset[0:5]: #REMOVE [BOUNDS] FOR WHOLE DATA SET   
    start = str(x['start_time'])
    end = str(x['end_time'])
    match_url = x['url'].split('=', 1)[1].split('&', 1)[0]
    name = x['clean_text']

    if(name in name_occurences):
        name_occurences[name] += 1
    else:
        name_occurences[name] = 0

    filename = name + '_' + str(name_occurences[name])
    #print(str(ct_loop) + "    " + filename)
    #ct_loop += 1
    crop(start, end, match_url + ".mp4", filename + ".mp4")
    
    
