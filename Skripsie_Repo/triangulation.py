import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size

file_data = cv.FileStorage()
file_data.open("StereoProperties.xml", cv.FILE_STORAGE_READ)

cameraMatrixR = file_data.getNode("Right_StereoMatrix").mat()
cameraMatrixL =file_data.getNode("Left_StereoMatrix").mat()

distortionR = file_data.getNode("Right_Distortion").mat()
distortionL =file_data.getNode("Left_Distortion").mat()

rotation =file_data.getNode("Rotation").mat()
translation =file_data.getNode("Translation").mat()

essential =file_data.getNode("Essential").mat()
fundamental =file_data.getNode("Fundamental").mat()

file_data.release()
print ("Stereo properties read succesfully!")
#####################testing project points parameters########################
objectPoints = np.zeros((7,3,1), np.float32)
imagePoints = np.random.random((1,2,1))
cameraMatrix = np.eye(3)
distCoeffs = np.zeros((5,1))
R1= np.eye(3)
T1 = np.zeros(3)
T1 = T1.astype(float)
R1 = R1.astype(float)
#R1 = cv.Rodrigues(R1)
#T1 = cv.Rodrigues(T1)
print (R1)
print (T1)
##############################################################################

###############################Preparing the projecton matrices############################
RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
RT2 = np.concatenate([rotation,translation], axis = -1)
projMatrix_R = cameraMatrixR@RT1 # projection matrix for camera 1
projMatrix_L = cameraMatrixL@RT2 # projection matrix for camera 2

########## chess board size
chessBoardSize = (14,9) #chess board width and height (blocks)
frameSize = (640,480) # frame size in pixels
objp = np.zeros((chessBoardSize[0]*chessBoardSize[1],3), np.float32)
objp[:,:2] = np.mgrid[0:chessBoardSize[0], 0:chessBoardSize[1]].T.reshape(-1,2)
leftCap = cv.VideoCapture(0,cv.CAP_DSHOW) # usb webcam
rightCap = cv.VideoCapture (1,cv.CAP_DSHOW) # built in camera 
leftCap.set (10, 100)
rightCap.set (10, 100)
print ("Succesfully opened video streams")

while True:
    isTrue_1, rightFrame = rightCap.read()
    isTrue_2, leftFrame = leftCap.read()
    rightFrame = cv.flip(rightFrame,1)
    leftFrame = cv.flip(leftFrame,1)

    #rightFrame = cv.undistort(rightFrame,cameraMatrixR,distortionR)
    #leftFrame = cv.undistort(leftFrame,cameraMatrixL,distortionL)
    rightFrame = cv.undistort(rightFrame,cameraMatrixR,distortionR)
    leftFrame = cv.undistort(leftFrame,cameraMatrixL,distortionL)

    rightFrame_gray = cv.cvtColor(rightFrame, cv.COLOR_BGR2GRAY)
    leftFrame_gray = cv.cvtColor(leftFrame, cv.COLOR_BGR2GRAY)

    # retR, cornersR = cv.findChessboardCorners(rightFrame_gray, chessBoardSize, None) # returns all the corners of the chessboard detected in the current grayscale image and stores it in 'corners'
    # retL, cornersL = cv.findChessboardCorners(leftFrame_gray, chessBoardSize, None) #CHECK findChessBoardCornersSB
    retR, cornersR, metaR = cv.findChessboardCornersSBWithMeta(rightFrame_gray, chessBoardSize,cv.CALIB_CB_LARGER+cv.CALIB_CB_MARKER) # returns all the corners of the chessboard detected in the current grayscale image and stores it in 'corners'
    retL, cornersL, metaL = cv.findChessboardCornersSBWithMeta(leftFrame_gray, chessBoardSize, cv.CALIB_CB_LARGER+cv.CALIB_CB_MARKER) #CHECK findChessBoardCornersSB   
    #print (cornersR)
    if retR == True and retL == True:
        #cornersR = cv.undistortPoints(cornersR,cameraMatrixR,distortionR)
        #cornersL = cv.undistortPoints(cornersL,cameraMatrixL,distortionL)
        cv.drawChessboardCorners(rightFrame, chessBoardSize, cornersR,retR)
        cv.drawChessboardCorners(leftFrame, chessBoardSize, cornersL,retL)
        # cornersR = np.reshape(cornersR,(1,n,2))
        # cornersL = np.reshape(cornersL,(1,n,2))
        print (cornersR)
        break
        # newCornersR,newCornersL = cv.correctMatches(fundamental,cornersR,cornersL)
        points4D=cv.triangulatePoints(projMatrix_R,projMatrix_L,cornersR,cornersL) ## check correct Matches
        points3D = points4D/points4D[-1]
        #print (points3D)
        points3D = points3D[ :-1]
        #points3D = cv.convertPointsFromHomogeneous(points4D)
        chesspoints = np.array([points3D[0],points3D[1], points3D[2]])
        #print(chesspoints)
        #print(chesspoints)
        reprojected2D,jacobian= cv.projectPoints(chesspoints,R1,T1,cameraMatrixR, distortionR)
        #print (reprojected2D)
        centre = (int(reprojected2D[59][0][0]),int (reprojected2D[59][0][1]))
        cv.circle (rightFrame,centre,5,(255,0,0), -1)
        #print (points3D)
        #print (str(points3D[0]) +"\t"+ str(points3D[1])+"\t"+ str(points3D[2]))
    cv.imshow("Right Camera",rightFrame)    
    cv.imshow("Left Camera",leftFrame)  

    k = cv.waitKey(1)
    if (k == 27):
        break  