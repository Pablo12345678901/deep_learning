import json # for json functions
import os # for exit codes
import re # regex
import requests # request to api
import sys # to get env var

def download_video_from_json(video_name, video_url, target_directory):
    print("Video name : " + video_name)
    print("Video url : " + video_url)
    print("Target directory : " + target_directory)
    print() # esthetic
    filepath = target_directory + "/" + video_name    
    print("Downloading video : %s"%video_name) 
    response = requests.get(video_url, stream = True)
    with open(filepath, 'wb') as f: 
        for chunk in response.iter_content(chunk_size = 1024*1024): 
            if chunk: 
                f.write(chunk)
    print("%s downloaded!\n"%filepath) 

def get_download_url(source_filepath, index):
    json_file = open(source_filepath, 'r')
    values = json.load(json_file)
    json_file.close()
    download_url = values['hits'][index]['urls']['mp4_download']
    return download_url

def get_pretty_json(obj):
    json_pretty_content = json.dumps(obj.json(), indent=4)
    return json_pretty_content

def get_video_name_without_special_char(source_filepath, index):
    json_file = open(source_filepath, 'r')
    values = json.load(json_file)
    json_file.close()
    video_name_with_special_char = values['hits'][index]['title']
    video_name_without_special_char = re.sub('[^A-Za-z0-9]+', '_', video_name_with_special_char)
    return video_name_without_special_char

def ping_url(url):
    # Ping the downloaded video before to do so is requested by API (see API doc)
    reponse = os.system("ping -c 1 " + url + " >/dev/null 2>&1 ")

def write_json_file(json_content, filepath):
    file_written = open(filepath, "w")
    file_written.write(json_content)
    file_written.close()



# Main variables
api_website = "https://api.coverr.co/videos"
number_of_videos_requested = 5
this_file = os.path.realpath(__file__)
project_directory= os.path.dirname(this_file)
print(project_directory)

# Get the API key from a file
api_key_filename = os.environ['REPERTOIRE_BACK_UP'] + "/coverr_api_key.txt"
api_key_file = open(api_key_filename, "r") 
api_key_from_file = api_key_file.read()
api_key_file.close()

# Request to API
parameters = {
    "api_key" : api_key_from_file,
    "page_size" : number_of_videos_requested,
    "query": "dogs",
    "urls" : "true",
}
# Parameters available
"""
page	Page number. Number. Default: 0
page_size	Number of videos per page. Number. Default: 20
query	Search videos by query. String. Default: '', see a note below
sort_by	How to sort videos. String. Valid values: date, popularity. Default: popularity
urls	Add urls property in the response. Boolean. Default: false
"""

# Call to api and save answer
response = requests.get(api_website, params=parameters)
file_with_response = project_directory + "/" + "ressources" + "/" + "json" + "/" + "response_from_api.json"
# Check the status code
if response.status_code == 200:
    # If okay, transform the content in a pretty way and write it into a file
    json_content = get_pretty_json(response)
    write_json_file(json_content, file_with_response) # back-up of request result
else :
    # Else exit with error code 1
    print("ERROR in the API call : returned value " + str(response.status_code))
    sys.exit(1)

# Download the videos
for i in range (0, number_of_videos_requested):
    video_name = get_video_name_without_special_char(file_with_response, i)
    video_url = get_download_url(file_with_response, i)
    target_directory = project_directory + "/" + "ressources" + "/" + "videos"
    ping_url(video_url)
    download_video_from_json(video_name, video_url, target_directory)