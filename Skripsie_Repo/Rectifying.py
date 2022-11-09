#########This calibrates both camera's. Computes the distortion parametres, corrects for lens distortion and rectifies the two cameras.
import cv2 as cv
import numpy as np
import stereoCameraCalibration
num = stereoCameraCalibration.num


################### read the stereoMap parametres###################################
print (num)

##########Starts the video stream from both cameras
leftCap = stereoCameraCalibration.leftCap
rightCap = stereoCameraCalibration.rightCap
##########

########## chess board size
chessBoardSize = (10,6) #chess board width and height (blocks)
frameSize = (640,480) # frame size in pixels
##########

##### Subpixel termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30,0.001) # criteria used for the termination when finding sub pixels in images 

########## Prepare object points
objp = np.zeros((chessBoardSize[0]*chessBoardSize[1],3), np.float32)
objp[:,:2] = np.mgrid[0:chessBoardSize[0], 0:chessBoardSize[1]].T.reshape(-1,2)
objp *= 23.5

########## preparing arrays to store the cooridantes of the real world and image planes
objPoints = [] #3D points in the real world
imgPointsL = [] #2D points in the image plane for the left camera
imgPointsR = [] #2D points in the image plane for the right camera
calibImagesL = [] #array that stores the calibration images for the left camera
calibImagesR = [] #array that stores the calibration images for the right camera

########## storing all the calibration images into the calibImages array, all images are converted into gray scale
for curr in range(1,num+1):
    currImgR = cv.imread('Photos\UNDISTORTED_StereoImages\\Right_Images\R_image_' + str(curr) + '.png')
    currImgL = cv.imread('Photos\UNDISTORTED_StereoImages\\Left_Images\L_image_' + str(curr) + '.png')
    grayImgR = cv.cvtColor(currImgR, cv.COLOR_BGR2GRAY)
    grayImgL = cv.cvtColor(currImgL, cv.COLOR_BGR2GRAY)
    calibImagesR.append(grayImgR)
    calibImagesL.append(grayImgL)
    cv.imshow("Gray Image Right " + str(curr),grayImgR)
    cv.waitKey(100)
    cv.imshow("Gray Image Left " + str(curr),grayImgL)
    cv.waitKey(100)



########## using a for loop to run through each gray image stored in the calibImages array
counter=0
for leftImage,rightImage in zip(calibImagesL, calibImagesR): 
    counter +=1
    L_frame = leftImage
    R_frame = rightImage
    # finding the chess board corners
    retR, cornersR = cv.findChessboardCorners(R_frame, chessBoardSize, None) # returns all the corners of the chessboard detected in the current grayscale image and stores it in 'corners'
    retL, cornersL = cv.findChessboardCorners(L_frame, chessBoardSize, None)
    # ret stores a boolean whether corners have been found or not
    #NB!!!!!! make sure that there is a thick white boarder around the chessboard for more robust algorithm searching
    # If corners have been found, add the object points and image points after some refining (using sub pixels)
    
    
    if retL and retR == True:
        objPoints.append (objp)
        subPixelCornersR = cv.cornerSubPix (R_frame, cornersR, (11,11),(-1,-1), criteria) # find sub pixel corners, using default parameters at the moment
        subPixelCornersL = cv.cornerSubPix (L_frame, cornersL, (11,11),(-1,-1), criteria)
        imgPointsR.append(subPixelCornersR)
        imgPointsL.append(subPixelCornersL)

        #Draw and display the corners found on the chessboard
        cv.drawChessboardCorners(R_frame, chessBoardSize, subPixelCornersR,retR)
        cv.drawChessboardCorners(L_frame, chessBoardSize, subPixelCornersL,retL)
        cv.imshow('Right image corners'+str(counter),R_frame)
        cv.waitKey(100)
        cv.imshow('Left image corners'+str(counter),L_frame)
        cv.waitKey(100)


cv.destroyAllWindows()


#################################33###### Calibrating the camera and getting camera matrix###################################################################
isTrueR, cameraMatrixR, distortionR, rvecsR, tvecsR = cv.calibrateCamera(objPoints, imgPointsR, frameSize, None, None)
heightR,widthR= R_frame.shape
optimalCameraMatrixR, roiR = cv.getOptimalNewCameraMatrix(cameraMatrixR,distortionR,(widthR,heightR),1,(widthR,heightR))

isTrueL, cameraMatrixL, distortionL, rvecsL, tvecsL = cv.calibrateCamera(objPoints, imgPointsL, frameSize, None, None)
heightL,widthL = L_frame.shape
optimalCameraMatrixL, roiL = cv.getOptimalNewCameraMatrix(cameraMatrixL,distortionL,(widthL,heightL),1,(widthL,heightL))
    
#################################################### Calibrating StereoVision#############################################################
flags=0
flags |=cv.CALIB_USE_INTRINSIC_GUESS  
# set these flags based on application
# this flag only fixes the intrinsic camera matrixes, therefore only the rotational, translational, essential and fundamental matrix 
# therefore the intrinsic paramtres are the same

criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

retStereo, newCameraMatrixR, distortionR, newCameraMatrixL, distortionL, rotation, translation, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(objPoints, imgPointsR, imgPointsL, optimalCameraMatrixR, distortionR, optimalCameraMatrixL, distortionL, calibImagesL[-1].shape[::-1], criteria_stereo, flags)

####################################### Stereo Rectification########################################
scale = -1
rectR, rectL, projMatrixR, projMatrixL, Q, roiR, roiL = cv.stereoRectify (newCameraMatrixR,distortionR,newCameraMatrixL,distortionL, calibImagesL[-1].shape[::-1],rotation, translation, flags=cv.CALIB_ZERO_DISPARITY,alpha = -1)
stereoMapL = cv.initUndistortRectifyMap(newCameraMatrixL,distortionL,rectL,projMatrixL, calibImagesL[-1].shape[::-1],cv.CV_16SC2)
stereoMapR = cv.initUndistortRectifyMap(newCameraMatrixR,distortionR,rectR,projMatrixR, calibImagesL[-1].shape[::-1],cv.CV_16SC2)

####################################Testing the rectification###############################################
while True:
    isTrue_1, rightFrame = rightCap.read()
    isTrue_2, leftFrame = leftCap.read()

    rectified_left = cv.remap(leftFrame,stereoMapL[0],stereoMapL[1],cv.INTER_LANCZOS4,cv.BORDER_CONSTANT,0)
    rectified_right = cv.remap(rightFrame,stereoMapR[0],stereoMapR[1],cv.INTER_LANCZOS4,cv.BORDER_CONSTANT,0)

    cv.imshow("Rectified Right Camera", rectified_right)
    cv.imshow("Rectified Left Camera", rectified_left)
    cv.imshow("Unrectified Right Camera", rightFrame)
    cv.imshow("Unrectified Left Camera", leftFrame)

    k = cv.waitKey(5)
    if (k == 27):
        break   

# #####################################Storing the mappings in an xml File##########################################
print ("Parameters Saved")
cv_file = cv.FileStorage ('stereoMap.xml', cv.FILE_STORAGE_WRITE)

cv_file.write("StereoMapL_x", stereoMapL[0])
cv_file.write("StereoMapL_y", stereoMapL[1])
cv_file.write("StereoMapR_x", stereoMapR[0])
cv_file.write("StereoMapR_y", stereoMapR[1])

cv_file.release()

rightCap.release()
leftCap.release()
print ('ALL GOOD!')
 
cv.waitKey(0)  
rightCap.release()
leftCap.release()
cv.destroyAllWindows() 