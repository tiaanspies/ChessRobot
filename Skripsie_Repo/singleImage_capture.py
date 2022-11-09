##########This takes photos from one camera's and stores in them in  a folder. This is used as a library for the calibration of a single  camera

import cv2 as cv
import numpy as np

########## Choose the camera to be calibrated##########
camera = 0 # 0 --> USB WebCam, 1--> Built in WebCam
camInfo =""
if camera == 0:
    camInfo = "USB WebCAM"
else:
    caminfo = "Built In WebCAM"    
capture = cv.VideoCapture(camera) 
########## 0 --> USB WebCam, 1--> Built in WebCam


num = 0

while capture.isOpened():
    isTrue, frame = capture.read()
    cv.imshow(camInfo, frame)
    k = cv.waitKey(5)

    if (k == 27):
        break
    elif k == ord ('s'):
        num +=1
        if camera==0:
            cv.imwrite('Photos\SingleImages\camUSB\Image_' + str(num) + '.png', frame)
            print (camInfo + " image has been saved")
        else:
            cv.imwrite('Photos\SingleImages\BuiltIn_cam\Image_' + str(num) + '.png', frame)  
            print (camInfo + " image has been saved")
        