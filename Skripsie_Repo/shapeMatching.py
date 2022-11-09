# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2 as cv
import numpy as np







#normal
# *Important
# *! Do not
# *?Check
# TODO: to do
#Video Stream-----

capture = cv.VideoCapture(2) # takes video stream from the first camera

def changeRes (width, height, brightness):
    capture.set (3, width)
    capture.set (4, height)
    capture.set (10, brightness)
changeRes(640 ,480,50)
#brightness was 50 ->worked good with a good light source
#-----



#Reading/preparing the marker image-----
marker = cv.imread ('Photos/circle_shape.jpg',0)
ret, thresh = cv.threshold(marker, 127, 255,0)
cv.imshow("marker",thresh)
contours,hierarchy = cv.findContours(thresh,2,1)
cnt1 = contours [0]

def empty(x):
    pass

#create my trackbar window
cv.namedWindow ("Selector")
#cv.resizeWindow ("Selector", 640,300)

cv.createTrackbar("Hue Min", "Selector",0,360,empty)
cv.setTrackbarPos("Hue Min", "Selector", 0)

cv.createTrackbar("Hue Max", "Selector",360,360,empty)
cv.setTrackbarPos("Hue Max", "Selector", 360)

cv.createTrackbar("Sat Min","Selector",0,255,empty)
cv.setTrackbarPos("Sat Min", "Selector", 0)

cv.createTrackbar("Sat Max","Selector",255,255,empty)
cv.setTrackbarPos("Sat Max", "Selector", 255)

cv.createTrackbar("Val Min","Selector",0,255,empty)
cv.setTrackbarPos("Val Min", "Selector", 0)

cv.createTrackbar("Val Max","Selector",255,255,empty)
cv.setTrackbarPos("Val Max", "Selector", 255)



while True:
    isTrue, frame = capture.read()
    frame = cv.flip(frame,1)
    output = frame.copy()
    #cv.imshow("Original Feed", output)
    blurred = cv.medianBlur(frame.copy(),3)
    #cv.imshow("Blurred", blurred)
    hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)

    h_min = int(cv.getTrackbarPos ("Hue Min","Selector"))
    h_max = int(cv.getTrackbarPos ("Hue Max","Selector"))
    S_min = int(cv.getTrackbarPos ("Sat Min","Selector"))
    S_max = int(cv.getTrackbarPos ("Sat Max","Selector"))
    V_min = int(cv.getTrackbarPos ("Val Min","Selector"))
    V_max = int(cv.getTrackbarPos ("Val Max","Selector"))
    lower_hsv = (h_min,S_min,V_min)
    upper_hsv = (h_max,S_max,V_max)
    mask1 = cv.inRange(hsv, lower_hsv, upper_hsv)
    #cv.imshow("Original Mask", mask1)
    mask2 = cv.erode(mask1, None, iterations=2)
    #cv.imshow("Eroded", mask2)
    mask3 = cv.dilate(mask2, None, iterations=2)
    cv.imshow("Final Mask", mask3)

    contours,hierachies = cv.findContours(mask3.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    center = None

    contour_list =[]
    coord = []
    if len(contours) > 0:
        for contour in contours:
            ret = cv.matchShapes(cnt1,contour,1,0.0)
            if (ret < 0.1):
                contour_list.append(contour)


        for currentCircle in contour_list:
            ((x,y),radius) = cv.minEnclosingCircle (currentCircle)
            M = cv.moments(currentCircle)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            coord.append(center)
            #cv.circle (frame,int(x), int(y), int(radius),(0,255,255),2)
            cv.circle(frame, center, 10, (0,0,255), -1)
            cv.circle(frame, (int(x), int(y)), 5, (255,255,255), -1)
            # if radius > 10:
	        #     cv.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
	        #     cv.circle(frame, center, 5, (0, 0, 255), -1) 

    if (len (coord) == 2):
        cv.line(frame,coord[0],coord[1],(0,255,0),3)
        cv.putText(frame,"Marker 1",coord[0], cv.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255))
        cv.putText(frame,"Marker 2",coord[1], cv.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255))
    elif (len(coord) == 3):
            pts = np.array([coord[0],coord[1],coord[2]], np.int32)
            pts = pts.reshape((-1,1,2))
            cv.polylines(frame, [pts],True,(0,255,0), 3)
            cv.putText(frame,"Marker 1",coord[0], cv.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255))
            cv.putText(frame,"Marker 2",coord[1], cv.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255))
            cv.putText(frame,"Marker 3",coord[2], cv.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255))
    elif (len(coord) == 4):
        pts = np.array([coord[0],coord[1],coord[2],coord[3]], np.int32)
        pts = pts.reshape((-1,1,2))
        cv.polylines(frame, [pts],True,(0,255,0), 3)
        cv.putText(frame,"Marker 1",coord[0], cv.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255))
        cv.putText(frame,"Marker 2",coord[1], cv.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255))
        cv.putText(frame,"Marker 3",coord[2], cv.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255))
        cv.putText(frame,"Marker 4",coord[3], cv.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255))


    cv.imshow("frame",frame)
    #cv.imshow ("Mask",mask)
    k = cv.waitKey(1) & 0xFF
    if (k == 27):
        break
    elif k == ord ('s'):
        cv.imwrite('Photos\SingleImages\Original_shape.png', output)
        cv.imwrite('Photos\SingleImages\Blurred_shape.png', blurred)
        cv.imwrite('Photos\SingleImages\Mask 1_shape.png', mask1)
        cv.imwrite('Photos\SingleImages\Mask 2_shape.png', mask2)
        cv.imwrite('Photos\SingleImages\Mask 3_shape.png', mask3)
        cv.imwrite('Photos\SingleImages\Frame_shape.png', frame)


capture.release()
cv.destroyAllWindows()    

