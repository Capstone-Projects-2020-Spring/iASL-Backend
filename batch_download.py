from pytube import YouTube, Stream
import json

#a subdirectory named "raw" must exist in the folder this script runs from
#NOTE: youtube changed their site recently so your pytube library must be slightly different for this to work
#      ask aidan for the fixed mixins.pyu

#Handling to get list of unique URLS from JSON
vid_urls = []
with open('MSASL_train.json', 'r') as f:
    dataset=json.load(f)
for x in dataset[0:5]: #REMOVE [BOUNDS] FOR WHOLE DATA SET
    cut_url = x['url'].split('&', 1)[0]
    if(cut_url not in vid_urls):
        print(cut_url)
        vid_urls.append(cut_url)

print("Number of videos:", len(vid_urls))

#Download all videos from URLs specified in vid_urls
for x in vid_urls:
    name = x.split('=')[1]
    YouTube(url=x).streams.first().download(output_path="raw/", filename=name)
