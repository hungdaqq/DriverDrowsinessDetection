import sys
import cv2
import numpy as np

# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade_left = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
eye_cascade_right = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')

def eye_tracker(v):
    # Open output file
    frameCounter = 0
    ret,frame = v.read()

    frameCounter = frameCounter + 1

    while(1):
        ret ,frame = v.read() # read another frame
        if ret == False:
            break

        eyes_left = eye_cascade_left.detectMultiScale(frame)
        eyes_right = eye_cascade_right.detectMultiScale(frame)

        for (ex1,ey1,ew1,eh1) in eyes_left:
            break
        for (ex,ey,ew,eh) in eyes_right:
            break
        
        # frame[ey+eh:ey, ex:ex1+ew1]
        croppedImg = frame[ey1:ey1+eh1, ex:ex1+ew1]
        if(croppedImg.shape[0]<=0 or croppedImg.shape[1]<=0):
            frameCounter = frameCounter + 1
            continue
        elif(croppedImg.shape[0] <= croppedImg.shape[1]/2.2):
            output_name = "./" + sys.argv[2] + str(frameCounter) + ".jpg"
            print(output_name)
            cv2.imwrite(output_name, croppedImg)
        cv2.rectangle(frame,(ex-5,ey-5),(ex+(ex1-ex)+ew1+5,ey+eh+5),(0,255,0),2)
        cv2.imshow('capture',frame)
        cv2.imshow('eye',croppedImg)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
        frameCounter = frameCounter + 1

if __name__ == '__main__':

    video = cv2.VideoCapture(int(sys.argv[1]))
    eye_tracker(video)