#Program to obtain OCR tokens for dedup image using Doctr	
#Link to the code: https://github.com/mindee/doctr


from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import os 

model = ocr_predictor(pretrained=True)

path_to_one_video = "/home/soumyajahagirdar/Desktop/NewsVideoQA/Dataset/Test/50_test_videos_non_dup/3/"

list_of_all_frames = os.listdir(path_to_one_video)

print("Total frames in the video: ", len(list_of_all_frames))

unimportant_words_list = ["CNNINEWS", "news18.com", "GNNNEWS", "GNN'NEWS", "V-GUARD", "WATER", "HEATERS", "18", ".", "CNN'NEWS", "GNNINEWS", "GNN/NEWS", "18.", "CNN/NEWS"]
frame_to_remove = []
for fr in list_of_all_frames:
	path = path_to_one_video + str(fr)
	single_img_doc = DocumentFile.from_images(path)
	result = model(single_img_doc)
	json_output = result.export()
	pages = json_output["pages"][0]
	ct = 0
	ocr_list = []
	for i in pages["blocks"]:
		for j in i["lines"]:
			ct+=1
			# print(j["words"][0]["value"])
			ocr_list.append(j["words"][0]["value"])
			# print("\n")
	for i in unimportant_words_list:
		if i in ocr_list:
			ocr_list.remove(i)

	print("Frame number", fr)
	print("OCR string:- ", ' '.join(ocr_list))
	print(len(ocr_list))

	if len(ocr_list) == 0:
		frame_to_remove.append(str(fr))


with open('/home/soumyajahagirdar/Desktop/NewsVideoQA/Dataset/Test/Unimportant_Test_Frames_List/not_good_frames_3.txt', 'w') as fp:
    for item in frame_to_remove:
        fp.write("%s\n" % item)
    print('Done')
# path = "/home/soumyajahagirdar/Desktop/NewsVideoQA/Dataset/Test/50_test_videos_non_dup/1/122.jpg"
# single_img_doc = DocumentFile.from_images(path)

# result = model(single_img_doc)

# json_output = result.export()
# pages = json_output["pages"][0]
# print((pages["blocks"][0].keys()))
# print(len(pages["blocks"][1]["words"]))
# print(len(pages["blocks"]))
# print(pages.keys())
# print((pages["blocks"][1]["lines"][0]["words"][0]["value"]))
# print(json_output.keys())









#-------------------------------------------------------------












# import os
# from doctr.models import ocr_predictor
# from doctr.io import DocumentFile
# from matplotlib.pyplot import plt

# model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)

# folder = "/home/soumyajahagirdar/Desktop/NewsVideoQA/Dataset/Test/50_test_videos_non_dup/1/"

# list_path_to_dedup_folder = os.listdir(folder)

# img_path = folder+str(list_path_to_dedup_folder[0])
# single_img_doc = DocumentFile.from_images(img_path)
# result = model(single_img_doc)
# result.show(single_img_doc)

# synthetic_pages = result.synthesize()
# plt.imshow(synthetic_pages[0]); plt.axis('off'); plt.show()
# # for i in list_path_to_dedup_folder:
# # 	img_path = folder+str(i)
# # 	single_img_doc = DocumentFile.from_images(img_path)
# # 	result = model(single_img_doc)
# # 	result.show(single_img_doc)
# # 	json_output = result.export()
# # 	print(json_output)
# # 	exit()