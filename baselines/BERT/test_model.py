from simpletransformers.question_answering import QuestionAnsweringModel
import json
import numpy as np
from functools import lru_cache
import json
import os
import logging
from collections import Counter

# logging.basicConfig(level=logging.INFO)
# transformers_logger = logging.getLogger("transformers")
# transformers_logger.setLevel(logging.WARNING)

# # predictions for each sample will be written to the "output_dir" specified below
# # see "predictions_test.json" in the specified output_dir for per sample predictions

'''
test files: 
1. Concatenation of all frames: each_video_ocr_test.json
2. Two frames of the particular second: three_ocr_test.json


'''



with open('/ssd_scratch/cvit/soumyajahagirdar/BERT_DATA/old_new_two_frame_camera_ready_bert_12_test.json') as f:
  json_data = json.load(f)


# # print((json_data[0]), json_data[1])
# # exit()

model_args = {"eval_batch_size": 1,  'output_dir': "/ssd_scratch/cvit/soumyajahagirdar/12fps_w_ft_old_sf/",  'doc_stride': 128, 'do_lower_case': True, 'max_seq_length': 384  }

# #Without finetuning
# model = QuestionAnsweringModel('bert', 'bert-large-cased-whole-word-masking-finetuned-squad', args=model_args, use_cuda=True)

# #With finetuning
model = QuestionAnsweringModel('bert', '/ssd_scratch/cvit/soumyajahagirdar/BERT/experiments/old_sf/', args=model_args, use_cuda=True)


model.eval_model(json_data)
prediction_file = open("/ssd_scratch/cvit/soumyajahagirdar/12fps_w_ft_old_sf/predictions_test.json")
pred_f = json.load(prediction_file)


# exit()
def lev_dist(a, b):
    
	@lru_cache(None)  # for memorization
	def min_dist(s1, s2):

		if s1 == len(a) or s2 == len(b):

			return len(a) - s1 + len(b) - s2
		if a[s1] == b[s2]:
			return min_dist(s1 + 1, s2 + 1)

		return 1 + min(
      min_dist(s1, s2 + 1),      # insert character
      min_dist(s1 + 1, s2),      # delete character
      min_dist(s1 + 1, s2 + 1),  # replace character
    )

	return min_dist(0, 0)


#************************************************************************************
# Works for single frame and full ocr
#For single frame
# acc = 0
# ls=0
# for k, v in pred_f.items():
# 	for j in json_data:
# 		idd = j["qas"][0]["id"]
# 		if str(idd) == str(k):
# 			pred_ans = v
# 			gt_ans = j["qas"][0]["answers"][0]["text"]

# 			if pred_ans.lower() == gt_ans.lower():
# 				acc+=1

# 			ss = lev_dist(gt_ans.lower(), pred_ans.lower())
# 			ss=1-(ss/max(len(gt_ans.lower()), len(pred_ans.lower())))

# 			if ss>=0.5:
# 				ss=ss
# 			else:
# 				ss=0
# 			ls+=ss

# print("Accuracy is: ", ((acc/max(len(json_data), len(pred_f)))*100))
# print("ANLS: ", ls/max(len(json_data), len(pred_f))*100)
# exit()


#*********************************************************************************************************

#For two frames

import statistics
from statistics import mode
from tqdm import tqdm

acc = 0
ls = 0

# print(len(pred_f))

# print(json_data[0])
# print(type(pred_f))
# exit()
all_ori_ids = []
all_gt_ans = []
for j in json_data:
	all_ori_ids.append(j["qas"][0]["original_id"])
	all_gt_ans.append(j["qas"][0]["answers"][0]["text"])

for i in tqdm(range(len(all_ori_ids))):
	all_ans = []
	gt_ans = all_gt_ans[i]
	for l in json_data:
		if str(all_ori_ids[i]) == str(l["qas"][0]["original_id"]):
			for k, v in pred_f.items():
				if str(l["qas"][0]["id"]) == str(k):
					all_ans.append(v)


	# print(all_ans, gt_ans)
	c = Counter(all_ans)
	voted_ans_l=(c.most_common(1))
	voted_ans = (voted_ans_l[0][0])
	# exit()
	# if str(gt_ans).lower() in all_ans:
	# 	voted_ans = str(gt_ans).lower()
	# else:
	# 	voted_ans = mode(all_ans) 
	
	# if str(voted_ans).lower() == str(gt_ans).lower():
	# 	acc+=1
	# print(gt_ans, voted_ans, all_ans)

	if str(gt_ans.lower()) == str(voted_ans).lower():
		acc+=1

	ss = lev_dist(str(gt_ans).lower(), str(voted_ans).lower())
    
    
	ss=1-(ss/max(len(gt_ans.lower()), len(str(voted_ans).lower())))

	if ss>=0.5:
		ss=ss
	else:
		ss=0
	ls+=ss

print("Accuracy is: ", ((acc/(len(json_data))*100)))
print("ANLS: ", ls/max(len(json_data), len(pred_f))*100)


exit()





#*********************************************************************************************

#For multiple frames

# c = Counter(all_que)

import statistics
from statistics import mode
from tqdm import tqdm

acc = 0
ls = 0

# print(len(pred_f))

# print(json_data[0])
# print(type(pred_f))
# exit()
all_ori_ids = []
all_gt_ans = []
for j in json_data:
	all_ori_ids.append(j["qas"][0]["original_id"])
	all_gt_ans.append(j["qas"][0]["answers"][0]["text"])

for i in tqdm(range(len(all_ori_ids))):
	all_ans = []
	gt_ans = all_gt_ans[i]
	for l in json_data:
		if str(all_ori_ids[i]) == str(l["qas"][0]["original_id"]):
			for k, v in pred_f.items():
				if str(l["qas"][0]["id"]) == str(k):
					all_ans.append(v)


	# print(all_ans, gt_ans)
	c = Counter(all_ans)
	voted_ans_l=(c.most_common(1))
	voted_ans = (voted_ans_l[0][0])
	# exit()
	# if str(gt_ans).lower() in all_ans:
	# 	voted_ans = str(gt_ans).lower()
	# else:
	# 	voted_ans = mode(all_ans) 
	
	# # if str(voted_ans).lower() == str(gt_ans).lower():
	# # 	acc+=1
	# print(gt_ans, voted_ans, all_ans)

	if str(gt_ans.lower()) == str(voted_ans).lower():
		acc+=1

	ss = lev_dist(str(gt_ans).lower(), str(voted_ans).lower())
    
    
	ss=1-(ss/max(len(gt_ans.lower()), len(str(voted_ans).lower())))

	if ss>=0.5:
		ss=ss
	else:
		ss=0
	ls+=ss

print("Accuracy is: ", ((acc/(len(json_data))*100)))
print("ANLS: ", ls/max(len(json_data), len(pred_f))*100)


exit()










#*********************************************************************************************
#check length of each data

# f1 = open("/ssd_scratch/cvit/soumyajahagirdar/BERT_data/closed_captions/cc_ocr_train.json")
# train_d = json.load(f1)

# f2 = open("/ssd_scratch/cvit/soumyajahagirdar/BERT_data/closed_captions/cc_ocr_val.json")
# val_d = json.load(f2)

# f3 = open("/ssd_scratch/cvit/soumyajahagirdar/BERT_data/closed_captions/cc_ocr_test.json")
# test_d = json.load(f3)


# print("Len of train data: ", len(train_d))
# print("Len of val data: ", len(val_d))
# print("Len of test data: ", len(test_d))

