# -*- coding: utf-8 -*-
"""
@author: Arti Sahu
"""

#importing the required libraries
import cv2
from mtcnn.mtcnn import MTCNN


#capture the video from video file
file_video_stream = cv2.VideoCapture('images/testing/modi.mp4')

#create an instance of MTCNN detector
mtcnn_detector = MTCNN()

#initialize the array variable to hold all face locations in the frame
all_face_locations = []

#loop through every frame in the video
while True:
    #get the current frame from the video stream as an image
    ret,current_frame = file_video_stream.read()
    #resize the current frame to 1/4 size to proces faster
    current_frame_small = cv2.resize(current_frame,(0,0),fx=0.25,fy=0.25)
    #detect all face locations using the mtcnn dectector
    all_face_locations = mtcnn_detector.detect_faces(current_frame_small)
    #looping through the face locations
    for index,current_face_location in enumerate(all_face_locations):
        #splitting the tuple to get the four position values of current face
        x,y,width,height = current_face_location['box']
        left_pos = x
        top_pos = y
        right_pos = x+width
        bottom_pos = y+height
        #change the position maginitude to fit the actual size video frame
        top_pos = top_pos*4
        right_pos = right_pos*4
        bottom_pos = bottom_pos*4
        left_pos = left_pos*4
        #printing the location of current face
        print('Found face {} at top:{},right:{},bottom:{},left:{}'.format(index+1,top_pos,right_pos,bottom_pos,left_pos))
        #draw rectangle around the face detected
        cv2.rectangle(current_frame,(left_pos,top_pos),(right_pos,bottom_pos),(0,0,255),2)
        #draw circles for every face keypoints
        keypoints = current_face_location['keypoints']
        cv2.circle(current_frame,(keypoints['left_eye'][0]*4,keypoints['left_eye'][1]*4),5,(0,255,0),1)
        cv2.circle(current_frame,(keypoints['right_eye'][0]*4,keypoints['right_eye'][1]*4),5,(0,255,0),1)
        cv2.circle(current_frame,(keypoints['nose'][0]*4,keypoints['nose'][1]*4),5,(0,255,0),1)
        cv2.circle(current_frame,(keypoints['mouth_left'][0]*4,keypoints['mouth_left'][1]*4),5,(0,255,0),1)
        cv2.circle(current_frame,(keypoints['mouth_right'][0]*4,keypoints['mouth_right'][1]*4),5,(0,255,0),1)
    #showing the current face with rectangle drawn
    cv2.imshow("Webcam Video",current_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#release the stream and cam
#close all opencv windows open
file_video_stream.release()
cv2.destroyAllWindows()        










