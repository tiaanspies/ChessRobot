# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2 as cv
import numpy as np







#Video Stream-----

capture = cv.VideoCapture(0) # takes video stream from the first camera

def changeRes (width, height, brightness):
    capture.set (3, width)
    capture.set (4, height)
    capture.set (10, brightness)
changeRes(640 ,480,100)
#brightness was 50 ->worked good with a good light source
#-----



#Reading/preparing the marker image-----
# marker = cv.imread ('Photos/orangeMarker.JPG')
# img = cv.resize(marker, (0,0), fx=0.2, fy=0.2)
# hsv = cv.cvtColor(marker, cv.COLOR_BGR2HSV)
# lower_range = (161, 100, 121)
# upper_range = (180, 194, 255) #tracking a red marker 

def empty(a):
    pass

#create my trackbar window
cv.namedWindow ("Selector")
cv.resizeWindow ("Selector", 640,300)

cv.createTrackbar("Hue Min", "Selector",0,360,empty)
cv.createTrackbar("Hue Max", "Selector",360,360,empty)
cv.createTrackbar("Sat Min","Selector",0,255,empty)
cv.createTrackbar("Sat Max","Selector",255,255,empty)
cv.createTrackbar("Val Min","Selector",0,255,empty)
cv.createTrackbar("Val Max","Selector",255,255,empty)



# create a mask for image
#mask = cv.inRange(hsv, lower_range, upper_range)

# display both the mask and the image side-by-side
#cv.imshow('mask',mask)
#cv.imshow('image', marker)


while True:
    isTrue, frame = capture.read()
    cv.imshow("Original", frame)
    frame = cv.flip(frame,1)
    output = frame.copy()
    #blurred = cv.medianBlur(frame,7)
    #b,g,r = cv.split(frame)
    blurred = cv.medianBlur(frame.copy(),3)
    hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)

    h_min = cv.getTrackbarPos("Hue Min","Selector")
    h_max = cv.getTrackbarPos("Hue Max","Selector")
    S_min = cv.getTrackbarPos("Sat Min","Selector")
    S_max = cv.getTrackbarPos("Sat Max","Selector")
    V_min = cv.getTrackbarPos("Val Min","Selector")
    V_max = cv.getTrackbarPos("Val Max","Selector")
    lower_hsv = (h_min,S_min,V_min)
    upper_hsv = (h_max,S_max,V_max)
    mask = cv.inRange(hsv, lower_hsv, upper_hsv)
    mask = cv.erode(mask, None, iterations=3)
    mask = cv.dilate(mask, None, iterations=3)

    # cv.imshow("frame",frame)
    # cv.imshow ("mask",mask)
    
    #mask = cv.inRange(hsv, lower_range, upper_range)
    # mask = cv.erode(mask, None, iterations=2)
    # mask = cv.dilate(mask, None, iterations=2)
    #canny = cv.Canny(mask,100,200)
    contours,hierachies = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #contours = imutils.grab_contours(contours)
    center = None
    #Detecting circles using the hough circles function---------
    gray = cv.cvtColor (output, cv.COLOR_BGR2GRAY)
#     circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1.2, 200) # takes frame, method, accumulator value , min distance
#    # ensure at least some circles were found
#     if circles is not None:
#         circles = np.round (circles[0,:]).astype("int")
#         for (x,y,r) in circles:
#             cv.circle (output, (x,y),r,(0, 255, 0), 4)
#             cv.rectangle(output, (x- 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
#-------------------

   #detecting for circles in the colour specified
    contour_list =[]
    coord = []
    if len(contours) > 0:


        for contour in contours:
            approx = cv.approxPolyDP(contour,0.01*cv.arcLength(contour,True),True)
            area = cv.contourArea(contour)
            if ((len(approx) > 10) & (area > 30)):
                contour_list.append(contour)


        for currentCircle in contour_list:
            ((x,y),radius) = cv.minEnclosingCircle (currentCircle)
            M = cv.moments(currentCircle)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            coord.append(center)
            if radius > 10:
	            cv.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
	            cv.circle(frame, center, 5, (0, 0, 255), -1) 
    #--end of detection of circles-------

    #draw lines between the two circles
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


    #this only computes the largest area------
        # c = max (contours,key = cv.contourArea)
        # ((x,y),radius) = cv.minEnclosingCircle (c)
        # M = cv.moments(c)
        # center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    #end of computing the largest area

        # if radius > 10:
	    #     cv.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
	    #     cv.circle(frame, center, 5, (0, 0, 255), -1) 

    cv.imshow("frame",frame)
    cv.imshow ("Mask",mask)
    #cv.imshow ("red",r)
    #cv.imshow("output", output)
    k = cv.waitKey(1) & 0xFF
    if (k == 27):
        break
    elif k == ord ('s'):
        cv.imwrite('Photos\SingleImages\Original_polygon_.png', output)
        cv.imwrite('Photos\SingleImages\mask_polygon_.png', mask)
        cv.imwrite('Photos\SingleImages\Frame_polygon_.png', frame)


capture.release()
cv.destroyAllWindows()    

