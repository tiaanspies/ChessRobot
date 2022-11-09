#########This calibrates both camera's. Computes the distortion parametres, corrects for lens distortion and rectifies the two cameras.
import cv2 as cv
import numpy as np

option =1

if option:
    import stereoImage_capture
    num = stereoImage_capture.num 
    ##########Starts the video stream from both cameras
    rightCap = stereoImage_capture.rightCapture
    leftCap = stereoImage_capture.leftCapture 
##########
else:
    num = 9
    ##########Starts the video stream from both cameras
    leftCap = cv.VideoCapture(1) ## usb webcam 1->left camera
    rightCap = cv.VideoCapture (0) ##0->right




########## chess board size
chessBoardSize = (14,9) #chess board width and height (blocks)
frameSize = (640,480) # frame size in pixels
##########

##### Subpixel termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30,0.001) # criteria used for the termination when finding sub pixels in images 



########## Prepare object points
objp = np.zeros((chessBoardSize[0]*chessBoardSize[1],3), np.float32)
objp[:,:2] = np.mgrid[0:chessBoardSize[0], 0:chessBoardSize[1]].T.reshape(-1,2)
objp *= 16 #23.5


########## preparing arrays to store the cooridantes of the real world and image planes
objPoints = [] #3D points in the real world
imgPointsL = [] #2D points in the image plane for the left camera
imgPointsR = [] #2D points in the image plane for the right camera
calibImagesL = [] #array that stores the calibration images for the left camera
calibImagesR = [] #array that stores the calibration images for the right camera

calibImagesL_colour = [] #array that stores the calibration images for the left camera in colour
calibImagesR_colour = [] #array that stores the calibration images for the right camera in colour

test_r = []
test_l = []

########## storing all the calibration images into the calibImages array, all images are converted into gray scale
for curr in range(1,num+1):
    currImgR = cv.imread('Photos\StereoImages\Right_Images\R_image_' + str(curr) + '.png')
    currImgL = cv.imread('Photos\StereoImages\Left_Images\L_image_' + str(curr) + '.png')
    calibImagesR_colour.append(currImgR)
    calibImagesL_colour.append(currImgL)

    test_r.append(currImgR)
    test_l.append(currImgL)
    
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
for leftImage,rightImage,leftImageColour,rightImageColour in zip(calibImagesL, calibImagesR,calibImagesL_colour,calibImagesR_colour): 
    counter +=1
    L_frame = leftImage
    R_frame = rightImage

    # finding the chess board corners
    retR, cornersR = cv.findChessboardCornersSB(R_frame, chessBoardSize, None) # returns all the corners of the chessboard detected in the current grayscale image and stores it in 'corners'
    retL, cornersL = cv.findChessboardCornersSB(L_frame, chessBoardSize, None) #CHECK findChessBoardCornersSB

    # retR, cornersR, metaR = cv.findChessboardCornersSBWithMeta(R_frame, chessBoardSize,cv.CALIB_CB_MARKER) # returns all the corners of the chessboard detected in the current grayscale image and stores it in 'corners'
    # retL, cornersL, metaL = cv.findChessboardCornersSBWithMeta(L_frame, chessBoardSize,cv.CALIB_CB_MARKER) #CHECK findChessBoardCornersSB   
    # # ret stores a boolean whether corners have been found or not
    #NB!!!!!! make sure that there is a thick white boarder around the chessboard for more robust algorithm searching
    # If corners have been found, add the object points and image points after some refining (using sub pixels)
    
    
    if retL and retR == True:
        objPoints.append (objp)
        subPixelCornersR = cv.cornerSubPix (R_frame, cornersR, (11,11),(-1,-1), criteria) # find sub pixel corners, using default parameters at the moment
        subPixelCornersL = cv.cornerSubPix (L_frame, cornersL, (11,11),(-1,-1), criteria)
        imgPointsR.append(subPixelCornersR)
        imgPointsL.append(subPixelCornersL)

        #Draw and display the corners found on the chessboard
        cv.drawChessboardCorners(rightImageColour, chessBoardSize, subPixelCornersR,retR)
        cv.drawChessboardCorners(leftImageColour, chessBoardSize, subPixelCornersL,retL)
        cv.imshow('Right image corners'+str(counter),rightImageColour)
        cv.waitKey(100)
        cv.imshow('Left image corners'+str(counter),leftImageColour)
        cv.waitKey(100)


#v.destroyAllWindows()
counter=0

########## Calibrating the camera and getting camera matrix
isTrueR, cameraMatrixR, distortionR, rvecsR, tvecsR = cv.calibrateCamera(objPoints, imgPointsR, frameSize, None, None) ##CHECK out calibrateCameraRO
print ("The type of rotational and translation data type is:")
heightR,widthR= R_frame.shape
print (f"RMSE for right camera:{isTrueR}")
optimalCameraMatrixR, roiR = cv.getOptimalNewCameraMatrix(cameraMatrixR,distortionR,(widthR,heightR),0,(widthR,heightR))

isTrueL, cameraMatrixL, distortionL, rvecsL, tvecsL = cv.calibrateCamera(objPoints, imgPointsL, frameSize, None, None)
heightL,widthL = L_frame.shape
print (f"RMSE for left camera:{isTrueR}")
optimalCameraMatrixL, roiL = cv.getOptimalNewCameraMatrix(cameraMatrixL,distortionL,(widthL,heightL),0,(widthL,heightL))


########## undistorting all the images
for leftImageColour,rightImageColour in zip(test_l,test_r): 
    counter +=1
    L_frame = leftImageColour
    R_frame = rightImageColour
    undistord_img_Right = cv.undistort(R_frame, cameraMatrixR,distortionR,None,optimalCameraMatrixR)
    undistord_img_Left = cv.undistort(L_frame, cameraMatrixL,distortionL,None,optimalCameraMatrixL)
    xR,yR,wR,hR = roiR
    xL,yL,wL,hL = roiL
    undistord_img_Right = undistord_img_Right[yR:yR+hR, xR:xR+wR]
    undistord_img_Left = undistord_img_Left[yL:yL+hL, xL:xL+wL]
    #cv.imshow("Undistored image Right " + str(counter), undistord_img_Right)
    #cv.imshow("Undistored image Left " + str(counter), undistord_img_Left)
    #cv.waitKey(100)

######################################Undistoring the video Streams######################################
num = 0


print ("Starting the undistorting")
while True:
    isTrue_1, rightFrame = rightCap.read()
    isTrue_2, leftFrame = leftCap.read()

    undistord_left = cv.undistort(leftFrame, cameraMatrixL,distortionL,None,optimalCameraMatrixL)
    xL,yL,wL,hL = roiL
    undistord_left = undistord_left[yL:yL+hL, xL:xL+wL]

    undistord_right = cv.undistort(rightFrame, cameraMatrixR,distortionR,None,optimalCameraMatrixR)
    xR,yR,wR,hR = roiR
    undistord_right = undistord_right[yR:yR+hR, xR:xR+wR]

    cv.imshow("Undistorted Right Camera", undistord_right)
    cv.imshow("Undistorted Left Camera", undistord_left)
    cv.imshow("distorted Right Camera", rightFrame)
    cv.imshow("distorted Left Camera", leftFrame)

    k = cv.waitKey(5)
    if (k == 27):

        break
    elif k == ord ('s'):
        num +=1
        cv.imwrite('Photos\StereoImages\Right_Images\R_image_' + str(num) + '.png', undistord_right)
        cv.imwrite('Photos\StereoImages\Left_Images\L_image_' + str(num)+ '.png', undistord_left)
        print ("Images have been saved")


################################################################################################
# f = open("cropping_properties", "w")
# f.write(str(xR))
# f.write(str(yR))
# f.write(str(wR))
# f.write(str(hR))
# f.write(str(xL))
# f.write(str(yL))
# f.write(str(wL))
# f.write(str(hL))
# f.close()
cv.destroyAllWindows()    

#################################################### Calibrating StereoVision#############################################################
flags=0
flags |=cv.CALIB_FIX_INTRINSIC
# set these flags based on application
# this flag only fixes the intrinsic camera matrixes, therefore only the rotational, translational, essential and fundamental matrix 
# therefore the intrinsic paramtres are the same

criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

retStereo, newCameraMatrixR, distortionR, newCameraMatrixL, distortionL, rotation, translation, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(objPoints, imgPointsR, imgPointsL, optimalCameraMatrixR, distortionR, optimalCameraMatrixL, distortionL, calibImagesL[-1].shape[::-1], criteria_stereo, flags)
print (f"The RMSE value for stereo calibration: {retStereo}")
#################NBNBNBNBNBNBNBNB#############################################################
#############################Saving camera properties to file###############################
cv_file = cv.FileStorage ('StereoProperties.xml', cv.FILE_STORAGE_WRITE)
cv_file.write("Right_StereoMatrix", newCameraMatrixR)
cv_file.write("Left_StereoMatrix",newCameraMatrixL)

cv_file.write("Right_Distortion", distortionR)
cv_file.write("Left_Distortion", distortionL)

cv_file.write("Rotation", rotation)
cv_file.write("Translation",translation)

cv_file.write("Essential", essentialMatrix)
cv_file.write("Fundamental",fundamentalMatrix)

cv_file.release()
####################################### Stereo Rectification########################################
# scale = 0.9
# rectR, rectL, projMatrixR, projMatrixL, Q, roiR, roiL = cv.stereoRectify (newCameraMatrixR,distortionR,newCameraMatrixL,distortionL, calibImagesL[-1].shape[::-1],rotation, translation, flags=cv.CALIB_ZERO_DISPARITY,alpha = -1)
# stereoMapL = cv.initUndistortRectifyMap(newCameraMatrixL,distortionL,rectL,projMatrixL, calibImagesL[-1].shape[::-1],cv.CV_16SC2)
# stereoMapR = cv.initUndistortRectifyMap(newCameraMatrixR,distortionR,rectR,projMatrixR, calibImagesL[-1].shape[::-1],cv.CV_16SC2)

# ####################################Testing the rectification###############################################
# while True:
#     isTrue_1, rightFrame = rightCap.read()
#     isTrue_2, leftFrame = leftCap.read()

#     rectified_left = cv.remap(leftFrame,stereoMapL[0],stereoMapL[1],cv.INTER_LANCZOS4,cv.BORDER_CONSTANT,0)
#     rectified_right = cv.remap(rightFrame,stereoMapR[0],stereoMapR[1],cv.INTER_LANCZOS4,cv.BORDER_CONSTANT,0)

#     cv.imshow("Rectified Right Camera", rectified_right)
#     cv.imshow("Rectified Left Camera", rectified_left)
#     cv.imshow("Unrectified Right Camera", rightFrame)
#     cv.imshow("Unrectified Left Camera", leftFrame)

#     k = cv.waitKey(5)
#     if (k == 27):
#         break   

# # #####################################Storing the mappings in an xml File##########################################
# print ("Parameters Saved")
# cv_file = cv.FileStorage ('stereoMap.xml', cv.FILE_STORAGE_WRITE)

# cv_file.write("StereoMapL_x", stereoMapL[0])
# cv_file.write("StereoMapL_y", stereoMapL[1])
# cv_file.write("StereoMapR_x", stereoMapR[0])
# cv_file.write("StereoMapR_y", stereoMapR[1])
# cv_file.release()

# print ('ALL GOOD!')
rightCap.release()
leftCap.release()
cv.destroyAllWindows() 

