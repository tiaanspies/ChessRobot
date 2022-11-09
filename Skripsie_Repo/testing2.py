import numpy as np
import cv2 as cv

photo = cv.imread('Photos\Convert2HSV.png')
photo = cv.cvtColor(photo,cv.COLOR_BGR2HSV)
cv.imshow("Converted model",photo)
cv.waitKey(0)
