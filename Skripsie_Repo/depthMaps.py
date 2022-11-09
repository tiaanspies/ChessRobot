import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
height = 480
width = 640
leftCapture = cv.VideoCapture(1) ## usb webcam
rightCapture = cv.VideoCapture (0) ##built in camera
leftCapture.set(3, width)
leftCapture.set(4, height)
rightCapture.set(3, width)
rightCapture.set(3, height)

num = 0
while leftCapture.isOpened():
    isTrue_1, rightImg = rightCapture.read()
    isTrue_2, leftImg = leftCapture.read()
    cv.imshow("Right Camera", rightImg)
    cv.imshow("Left Camera", leftImg)
    k = cv.waitKey(5)

    if (k == 27):
        break
    elif k == ord ('s'):
        num +=1
        cv.imwrite('Photos\Right_Images\R_image_' + str(num) + '.png', rightImg)
        cv.imwrite('Photos\Left_Images\L_image_' + str(num)+ '.png', leftImg)
        print ("Images have been saved")
        

cv.destroyAllWindows()
right_depth = cv.imread('Photos\Right_Images\R_image_'+str(num) + '.png'  ,cv.IMREAD_GRAYSCALE)
left_depth = cv.imread('Photos\Left_Images\L_image_'+str(num) + '.png' , cv.IMREAD_GRAYSCALE)
cv.imshow("Right",right_depth)
cv.imshow ("Left",left_depth)
stereo = cv.StereoBM_create(numDisparities=0,blockSize=5)
depth = stereo.compute(right_depth, left_depth)
plt.imshow(depth)
plt.axis('off')
plt.show()


