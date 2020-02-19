from pytube import YouTube, Stream
import json

#a subdirectory named "raw" must exist in the folder this script runs from

#Handling to get list of unique URLS from JSON
vid_urls = []
with open('MSASL_train.json', 'r') as f:
    dataset=json.load(f)
for x in dataset[0:100]: #REMOVE [BOUNDS] FOR WHOLE DATA SET
    cut_url = x['url'].split('&', 1)[0]
    if(cut_url not in vid_urls):
        print(cut_url)
        vid_urls.append(cut_url)

print("Number of videos:", len(vid_urls))

#Download all videos from URLs specified in vid_urls
num_unavailable = 0
num_downloaded = 0
for x in vid_urls:
    name = x.split('=')[1]
    try:
        YouTube(url=x).streams.first().download(output_path="raw/", filename=name)
        num_downloaded += 1
    except Exception:
        print("Video " + name + " is unavailable for download.")
        num_unavailable += 1

print(str(num_downloaded) + " videos successfully downloaded.")
print(str(num_unavailable) + " video(s) could not be downloaded.")
