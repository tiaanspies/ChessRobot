import numpy as np
import cv2 as cv
import os
import time
from matplotlib import pyplot as plt
def empty(a):
    pass

def main():
    # def videoStuff(low):
    #     detected_edges = cv.Canny(blur, low, low*2, 3)
    #     arr = np.array(detected_edges)
    #     cv.imshow('frame', arr)

    boardSizeInt = (7, 7)
    end = 0

    # Create trackbars for calibrating color
    # cv.namedWindow('Selector')
    # cv.resizeWindow ("Selector", 640,300)
    # cv.createTrackbar("Hue Min", "Selector", 0, 360, empty)
    # cv.createTrackbar("Hue Max", "Selector", 360, 360, empty)
    # cv.createTrackbar("Sat Min","Selector", 0, 255, empty)
    # cv.createTrackbar("Sat Max","Selector", 255, 255, empty)
    # cv.createTrackbar("Val Min","Selector", 0, 255, empty)
    # cv.createTrackbar("Val Max","Selector", 255, 255, empty)


    # Our operations on the frame come here
    frame = cv.imread('IMG_0165.png')
    frame = cv.resize(frame, (640, 480), interpolation=cv.INTER_LANCZOS4)
    # frame = cv.imread('istockphoto-806894546-612x612.jpg')

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # blur = cv.blur(gray, (3, 3))
    blur = gray
    ret, th3 = cv.threshold(blur, 137, 255, cv.THRESH_BINARY_INV)
    
    element = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    erode = cv.erode(th3, element)
    dilate = cv.dilate(erode, element)
    # hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)

    success, corners = cv.findChessboardCorners(dilate, boardSizeInt, cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE)
    # corners = cv.goodFeaturesToTrack(frame, 100, 0.4, 10)
    
    if success:
        print("FOUND IT!!!!")
        # for corner in corners:
        #     corners_np = np.int0(corner)
        #     x, y = np.ravel(corners_np)

        #     # frame = cv.circle(frame, (x, y), 5, (0, 255, 0), -2)
        frame = cv.drawChessboardCorners(frame, boardSizeInt, corners, patternWasFound=success)
    
    cv.imshow('image', th3)
    cv.imshow('frame', frame)
    cv.imshow('erode', erode)
    cv.imshow('dialate', dilate)
    cv.waitKey(-1)

    


if __name__ == "__main__":
    main()