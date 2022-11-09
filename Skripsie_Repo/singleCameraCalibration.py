##########This calibrates only for a single camera. In this case it will only calibrate and compute the camera matrix for the USB web cam

import cv2 as cv
import sys
import numpy as np
import glob
import singleImage_capture
np.set_printoptions(threshold=sys.maxsize)

from numpy.core.fromnumeric import reshape
########## destroys all the windows from the image_calibration.py script for easier viewing
cv.destroyAllWindows()
##########
imgNum = singleImage_capture.num 
camera = singleImage_capture.camera

#########chess board size
CBsize = (10,6) #chess board width and height (blocks)
frameSize = (640,480) # frame size in pixels

#####termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30,0.001) # criteria used for the termination when finding sub pixels in images 
#(finding the corners of sub pixels to increase accuracy), find exact corner locations
##NB!!! look up what sub pixels are and why we use them!!!!

########## prepare object points
objp = np.zeros((CBsize[0]*CBsize[1],3), np.float32)
objp[:,:2] = np.mgrid[0:CBsize[0], 0:CBsize[1]].T.reshape(-1,2)
print (objp)
objp *= 23.5

########## preparing arrays to store the cooridantes of the real world and images\
objPoints = []
imgPoints = []
calibImages = [] #array that stores the calibration images

########## storing all the calibration images into the calibImages array, all images are converted into gray scale
for num in range(1,imgNum+1):
    if camera ==0:
        currImg = cv.imread('Photos\SingleImages\camUSB\image_' + str(num) + '.png')
    else:
        currImg = cv.imread('Photos\SingleImages\BuiltIn_cam\image_' + str(num) + '.png')    

    grayImg = cv.cvtColor(currImg, cv.COLOR_BGR2GRAY)
    calibImages.append(grayImg)
    cv.imshow("Gray Image " + str(num),grayImg)
    cv.waitKey(1000)



########## using a for loop to run through each gray image stored in the calibImages array
counter=0
for image in calibImages:
    counter +=1
    # finding the chess board corners
    ret, corners = cv.findChessboardCorners(image, CBsize, None) # returns all the corners of the chessboard detected in the current grayscale image and stores it in 'corners'
    # ret stores a boolean whether corners have been found or not
    #NB!!!!!! make sure that there is a thick white boarder around the chessboard for more robust algorithm searching
    # If corners have been found, add the object points and image points after some refining (using sub pixels)
    if ret == True:
        objPoints.append (objp)
        subPixelCorners = cv.cornerSubPix (image, corners, (11,11),(-1,-1), criteria) # find sub pixel corners, using default parameters at the moment
        imgPoints.append(corners)

        #Draw and display the corners found on the chessboard
        cv.drawChessboardCorners(image, CBsize, subPixelCorners,ret)
        cv.imshow('corners of image ' + str(counter),image)
        cv.waitKey(1000)


# cv.destroyAllWindows()


########## Calibrating the camera and getting camera matrix
isTrue, cameraMatrix, distortion, rvecs, tvecs = cv.calibrateCamera(objPoints, imgPoints, frameSize, None, None)
print ("Camera Calibrated: ",isTrue)
print ("Camera Matrix: ", cameraMatrix)
print ("Distortion Parametres: ", distortion)
print ("Rotation Vectors: ",rvecs)
print ("Translation Vectors: ", tvecs)

########## undistorting all the images
for num in range(1,imgNum+1):

    if camera == 0:
        img_calib = cv.imread('Photos\SingleImages\camUSB\image_'+str(num)+'.png')
    else:
        img_calib = cv.imread('Photos\SingleImages\BuiltIn_cam\image_'+str(num)+'.png')

    height,width = img_calib.shape[0:2]
    optimalCameraMatrix,roi = cv.getOptimalNewCameraMatrix(cameraMatrix,  distortion, frameSize,1,frameSize)

    undistord_img = cv.undistort(img_calib, cameraMatrix,distortion,None,optimalCameraMatrix)
    x,y,w,h = roi
    undistord_img = undistord_img[y:y+h, x:x+w]
    cv.imshow("Undistored image " + str(num), undistord_img)
    cv.waitKey(1000)


# mapx, mapy  =cv.initUndistortRectifyMap(cameraMatrix, distortion, None, optimalCameraMatrix, (width,height),5)
# dst = cv.remap(img_calib, mapx, mapy, cv.INTER_LINEAR)
# x,y,w,h = roi
# dst = dst[y:y+h, x:x+w]
# cv.imshow("Undistored image _2", dst)
cv.waitKey(0)    
