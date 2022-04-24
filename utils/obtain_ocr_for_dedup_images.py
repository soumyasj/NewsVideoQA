import imp
import os, io
from traceback import print_tb
from google.cloud import vision
from google.cloud.vision_v1 import types
import json
import pickle

"""
The following are the videos for which OCR tokens are obtained:
1. 120


"""

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/home/soumyajahagirdar/Desktop/GoogleOCR/venv/visionapi/visionapi_key.json"


all_frames_info = {}


all_frames = os.listdir("/home/soumyajahagirdar/Desktop/NewsVideoQA/Dataset/Test/50_test_videos_non_dup/120/")

all_frames_listter = []

ct = 0

for frame_id in all_frames:
	if ct%20 == 0:
		print(ct)
	ct+=1
	each_frame_info_list = []
	vid_number = 120
	each_frame_info_list.append(vid_number)
	each_frame_info_list.append(frame_id)
	
	file_name = "/home/soumyajahagirdar/Desktop/NewsVideoQA/Dataset/Test/50_test_videos_non_dup/120/" + frame_id
	# exit()

	with io.open(file_name, 'rb') as image_file:
	    content = image_file.read()

	image = vision.Image(content=content)
	client = vision.ImageAnnotatorClient()
	response = client.text_detection(image=image)
	labels = response.text_annotations
	if len(labels) > 0:
	    text = labels[0].description
	else:
	    text = ""

	each_word_list = []

	for i in range(len(labels)):

		each_label = labels.pop()
		each_word = each_label.description
		co_list = []
		for i in range(len(each_label.bounding_poly.vertices)):
			x = str(each_label.bounding_poly.vertices.pop())
			try:
				x_list = (x.split("\n"))
				x_co = int(x_list[0].split(":")[1])
				y_co = int(x_list[1].split(":")[1])
				co_list.append([x_co, y_co])
			except:
				print("out of range")

		l=[]
		l.append(each_word)
		l.append(co_list)

		each_word_list.append(l)
	
	each_frame_info_list.append((each_word_list))

	all_frames_listter.append(each_frame_info_list)


final_dict = {}

final_dict["data"] = all_frames_listter
final_dict["dataset_name"] = "NewsVideoQA"
final_dict["dataset_version"] = "trial_0.1"


with open("ocr_info_vid_no_120.json", "w") as outfile:
    json.dump(final_dict, outfile)


print("File written successfully!")