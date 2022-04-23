#Program to convert videos to frames and store them

import cv2 
import os
import pandas as pd


def main(video_numbers, input_video_path, output_video_frame_path):
   
    for i in video_numbers:
        ip_video_path = (input_video_path+str(i))
        output_frame_path = (output_video_frame_path+str(i))
        isExist = os.path.exists(output_frame_path)
        if not isExist:
            os.makedirs(output_frame_path)

        video_name = ip_video_path + "/" + (os.listdir(ip_video_path)[0])
        video = cv2.VideoCapture(video_name)
        ct=0

        while(True): 
            ct=ct+1
            if ct%100==0:
                print(ct)
            ret,frame = video.read() 
            if ret: 
                name = output_frame_path + "/"+str(ct)+'.jpg'
                cv2.imwrite(name, frame) 
            else: 
                break 
        

if __name__ == "__main__":
    video_number_path = "../Dataset/Test/test_video_link.csv"
    video_number_d = pd.read_csv(video_number_path)
    video_numbers = list(video_number_d["Number in Complete dataset"])

    input_video_path = "../Dataset/Test/50_test_videos/"
    output_video_frame_path = "../Dataset/Test/50_test_videos_frames/"

    main(video_numbers, input_video_path, output_video_frame_path)