import numpy as np
import cv2 as cv
import os
import time

def empty(a):
    pass

def main():
    # def videoStuff(low):
    #     detected_edges = cv.Canny(blur, low, low*2, 3)
    #     arr = np.array(detected_edges)
    #     cv.imshow('frame', arr)

    boardSizeInt = (7, 7)
    end = 0
 
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # Create trackbars for calibrating color
    cv.namedWindow('Selector')
    cv.resizeWindow ("Selector", 640,400)
    cv.createTrackbar("Hue Min", "Selector", 0, 360, empty)
    cv.createTrackbar("Hue Max", "Selector", 360, 360, empty)
    cv.createTrackbar("Sat Min","Selector", 0, 255, empty)
    cv.createTrackbar("Sat Max","Selector", 255, 255, empty)
    cv.createTrackbar("Val Min","Selector", 0, 255, empty)
    cv.createTrackbar("Val Max","Selector", 255, 255, empty)
    cv.createTrackbar("Thresh Min","Selector", 0, 255, empty)
    cv.createTrackbar("Erode Size","Selector", 1, 20, empty)
    cv.createTrackbar("Area min","Selector", 0, 10000, empty)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Our operations on the frame come here       

        h_min = cv.getTrackbarPos ("Hue Min","Selector")
        h_max = cv.getTrackbarPos ("Hue Max","Selector")
        S_min = cv.getTrackbarPos ("Sat Min","Selector")
        S_max = cv.getTrackbarPos ("Sat Max","Selector")
        V_min = cv.getTrackbarPos ("Val Min","Selector")
        V_max = cv.getTrackbarPos ("Val Max","Selector")
        T_min = cv.getTrackbarPos ("Thresh Min","Selector")
        erode_size = cv.getTrackbarPos ("Erode Size","Selector")
        cont_area = cv.getTrackbarPos ("Area min","Selector")

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blur = cv.blur(gray, (3,3))

        ret, th3 = cv.threshold(blur, T_min, 255, cv.THRESH_BINARY_INV)
        
        element = cv.getStructuringElement(cv.MORPH_RECT, (2*erode_size+1, 2*erode_size+1))
        erode = cv.erode(th3, element)
        dilate = cv.dilate(erode, element)
        
        # th3 = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,21,1)
        
        # hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # canny = cv.Canny(gray, T_min, T_max)
        # img_contour = canny
        # getContours(canny, img_contour, cont_area)
        
        lower_hsv = (h_min, S_min, V_min)
        upper_hsv = (h_max, S_max, V_max)

        # mask = cv.inRange(hsv, lower_hsv, upper_hsv)

        found_corners, corners = cv.findChessboardCorners(dilate, boardSizeInt)
        cv.drawChessboardCorners(frame, boardSizeInt, corners, found_corners)
        # corners = cv.goodFeaturesToTrack(frame, 100, 0.4, 10)
        
        
        # Find Frame time
        # start = end
        # end = time.time()

        os.system('cls')
        # print(mask.shape)
        # print(end-start)

        # Display the resulting frame
        
        # cv.imshow('gray', gray)
        cv.imshow('blur', blur)
        cv.imshow('erode', erode)
        cv.imshow('dilate', dilate)
        cv.imshow('thresh', th3)
        cv.imshow('frame', frame)


        if cv.waitKey(1) == ord('q'):
            break
    # When everything done, release the capture

    cap.release()
    cv.destroyAllWindows()

def getContours(img, imgcontours, thresh):
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    
    for cont in contours:
        area = cv.contourArea(cont)

        if area > thresh:
            print(area)
            cv.drawContours(imgcontours, cont, -1, (255, 0, 0), 7)


if __name__ == "__main__":
    main()