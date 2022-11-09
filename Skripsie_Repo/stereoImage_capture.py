##########This takes photos from both camera's and stores in them in their seperate folder. This is used as a library for the calibration of both the camera's

import cv2 as cv
import numpy as np

leftCapture = cv.VideoCapture(1) ## usb webcam
rightCapture = cv.VideoCapture (0) ##built in camera
width = 640
height = 480

# leftCapture.set (3, width)
# leftCapture.set (4, height)

# rightCapture.set (3, width)
# rightCapture.set (4, height)


num = 0
while leftCapture.isOpened():
    isTrue_1, rightImg = rightCapture.read()
    isTrue_2, leftImg = leftCapture.read()
    rightImg = cv.flip(rightImg,1)
    leftImg = cv.flip(leftImg,1)
    cv.imshow("Right Camera", rightImg)
    cv.imshow("Left Camera", leftImg)
    k = cv.waitKey(5)

    if (k == 27):
        break
    elif k == ord ('s'):
        num +=1
        cv.imwrite('Photos\StereoImages\Right_Images\R_image_' + str(num) + '.png', rightImg)
        cv.imwrite('Photos\StereoImages\Left_Images\L_image_' + str(num)+ '.png', leftImg)
        print ("Images have been saved")

cv.destroyAllWindows()
        