import cv2

import numpy as np

win = cv2.namedWindow('dzd',cv2.WINDOW_NORMAL)

cv2.resizeWindow('dzd',640,200)

#打开摄像头

v = cv2.VideoCapture(0)

# v =cv2.VideoCapture('./dzd2.mp4')

win = cv2.namedWindow('dzd',cv2.WINDOW_NORMAL)

cv2.resizeWindow('dzd',1000,680)
7
face_detector = cv2.CascadeClassifier('./haarcascade_frontalface_alt2.xml')

#视频是由一张张图片组成，一张就是一帧

num=1

while True:

    flag,frame=v.read()

    if not flag:

        break

    frame=cv2.resize(frame,(400,400))

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    face_zones = face_detector.detectMultiScale(gray,scaleFactor = 1.1,

    minNeighbors = 3)

    for x,y,w,h in face_zones[:1]:

        cv2.rectangle(frame,pt1 = (x,y),pt2 = (x+w,y+h),color = [0,0,255],thickness=2)

        cv2.imshow('dzd',frame)

    #是否保存捕获到的头像

    # if len(face_zones)>=1:

    # f=input('是否保存这张图片？Y/N')

    # if f=='Y':

    # face=frame[y+2:y+h+1,x+2:x+w-1]

    # face=cv2.resize(face,(100,100))

    # cv2.imwrite('./%d.jpg'%(num),face)

    # num+=1

    # if f=='t':

    # break

        key=cv2.waitKey(1)

        if key==ord('q'):

            break

        v.release()

        cv2.destroyAllWindows()