import numpy as np
import cv2 as cv
from numpy.core.fromnumeric import size
from scipy.ndimage.measurements import label
import trackingFunctions
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from itertools import count
from matplotlib.animation import FuncAnimation
from scipy.signal import savgol_filter
from scipy.signal import lfilter
import scipy.io


def empty(x):
    pass


#####################Preparing marker for shpape matching##########################
shape = trackingFunctions.prepareMatching('Photos/circle_shape.jpg')
frameSize = (640,480) # frame size in pixels
distances = np.zeros(3)
referencePoint = np.zeros(3)
index = count()
startGraph =0 #used for starting the graphing measurements for test 1
allDistPR = [] # used to store all the distnaces tracked for distance PR
allDistPQ = [] # used to store all the distnaces tracked for distance PQ
allDistRQ = [] # used to store all the distnaces tracked for distance RQ
test = 2     #1->Distance between markers, #2->Distance between Z and reference, #3-V speed measurements #4 error model
prevTime = 0 #seconds
currTime = 0 #seconds
prevpoints3D = np.zeros((3,3))
timeCounter =0
dist =[0]
speedArray=[]
xxValues = []
z_fluctuation = []
plt.style.use('fivethirtyeight')

#######################error modeling##########################
allXp = []
allYp = []
allZp = []
allXq = []
allYq = []
allZq = []
allXr = []
allYr = []
allZr = []
#######################error modeling##########################

############################################Stereo Parameters##############################################################



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

###############################Preparing the projecton matrices############################
RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
R1= np.eye(3)
T1 = np.zeros(3)
T1 = T1.astype(float)
R1 = R1.astype(float)
RT2 = np.concatenate([rotation,translation], axis = -1)
projMatrix_R = cameraMatrixR@RT1 # projection matrix for camera 1
projMatrix_L = cameraMatrixL@RT2 # projection matrix for camera 2
##############################NB#######################################
# add error calculator so that only stereo properties with a certain error can be used
#######################################################################Starts the video stream from both cameras
leftCap = cv.VideoCapture(1,cv.CAP_DSHOW) # usb webcam 0>left camera
rightCap = cv.VideoCapture (0,cv.CAP_DSHOW) # built in camera 1>right camera
leftCap.set (10, 100)
rightCap.set (10, 100)
print ("Succesfully opened video streams")
#########################################

##########Creating trackbars#########################
trackingFunctions.createTrackbars("Right Mask")
trackingFunctions.createTrackbars("Left Mask")
# ####################################################
# trackingFunctions.threshTrackBar("Right Mask")
# trackingFunctions.threshTrackBar("Left Mask")
#################################################
rightHSV=[]
leftHSV=[]
centre = []
while True:
    isTrue_1, rightFrame = rightCap.read()
    isTrue_2, leftFrame = leftCap.read()
    rightFrame = cv.flip(rightFrame,1)
    leftFrame = cv.flip(leftFrame,1)

    rightFrame = cv.undistort(rightFrame,cameraMatrixR,distortionR)
    leftFrame = cv.undistort(leftFrame,cameraMatrixL,distortionL)
    #######undistort points only using undistortPoints() <------ look up function
#############################
    #Get HSV colour range for mask
    rightHSV= trackingFunctions.getRangeHSV("Right Mask")
    leftHSV= trackingFunctions.getRangeHSV("Left Mask")
    rightMask = trackingFunctions.createMask(rightFrame,rightHSV)
    leftMask = trackingFunctions.createMask(leftFrame,leftHSV)

#############################    
    # threshVal_right = cv.getTrackbarPos("threshVal","Right Mask")
    # threshVal_left = cv.getTrackbarPos("threshVal","Left Mask")
    # rightMask = trackingFunctions.thresholdMask(rightFrame,threshVal_right)
    # leftMask = trackingFunctions.thresholdMask(leftFrame,threshVal_left)
##############################    

    rightMarkers = trackingFunctions.findMarkers(rightMask,shape)
    leftMarkers = trackingFunctions.findMarkers(leftMask,shape)

    trackingFunctions.drawMarkers(rightMarkers,rightFrame)
    trackingFunctions.drawMarkers(leftMarkers,leftFrame)

    cv.imshow("Right Mask ", rightMask)
    cv.imshow("Left Mask ", leftMask)
    cv.imshow("Right Camera", rightFrame)
    cv.imshow("Left Camera", leftFrame)


    k = cv.waitKey(5)
    if (k == 27):
        break
    

cv.destroyAllWindows() 
cv.namedWindow ("Z_window")
cv.createTrackbar("Z_distance", "Z_window",0,200,empty)
cv.setTrackbarPos("Z_distance", "Z_window", 0)
while True:
    k = cv.waitKey(5)
    Z_val = cv.getTrackbarPos("Z_distance","Z_window")
    isTrue_1, rightFrame = rightCap.read()
    isTrue_2, leftFrame = leftCap.read()
    rightFrame = cv.flip(rightFrame,1)
    leftFrame = cv.flip(leftFrame,1)
    rightFrame = cv.undistort(rightFrame,cameraMatrixR,distortionR)
    leftFrame = cv.undistort(leftFrame,cameraMatrixL,distortionL)
#####################################################    
    rightMask = trackingFunctions.createMask(rightFrame,rightHSV)
    leftMask = trackingFunctions.createMask(leftFrame,leftHSV)
    mask1 = cv.medianBlur(rightFrame,3)
    mask2 = cv.cvtColor(mask1, cv.COLOR_BGR2HSV)
    mask3 = cv.inRange(mask2, rightHSV[0], rightHSV[1])
    mask4 = cv.erode(mask3, None, iterations=5)
    mask5 = cv.dilate(mask4, None, iterations=5)
####################################################

    # rightMask = trackingFunctions.thresholdMask(rightFrame,threshVal_right)
    # leftMask = trackingFunctions.thresholdMask(leftFrame,threshVal_left)
#######################################################

    if test ==2:
    ##########Drawing the reference point on the image frames##########
        projectedReferenceR,jacobian= cv.projectPoints(referencePoint,R1,T1,cameraMatrixR, distortionR,Z_val)
        projectedReferenceL,jacobian= cv.projectPoints(referencePoint,rotation,translation,cameraMatrixL, distortionL,Z_val)
        projectedReferenceR = projectedReferenceR.astype(int)
        projectedReferenceL = projectedReferenceL.astype(int)
    ##########Drawing the reference point on the image frames##########

    rightMarkers = trackingFunctions.findMarkers(rightMask,shape)
    leftMarkers = trackingFunctions.findMarkers(leftMask,shape)

    trackingFunctions.drawMarkers(rightMarkers,rightFrame)
    trackingFunctions.drawMarkers(leftMarkers,leftFrame)

    trackingFunctions.drawMarkers(rightMarkers,rightFrame)
    trackingFunctions.drawMarkers(leftMarkers,leftFrame)

    points3D = trackingFunctions.get3Dpoints(rightMarkers,leftMarkers,fundamental,projMatrix_R,projMatrix_L)
    rightPoints = np.array(rightMarkers)
    leftPoints = np.array(leftMarkers)
    if len(rightPoints)==3 and len(leftPoints)==3:
        centre1,centre2,centre3,centre4R = trackingFunctions.projectPoints(points3D,R1,T1,cameraMatrixR,distortionR,Z_val) #I willl have to change this
        cv.circle (rightFrame,centre1,2,(0,255,0), -1)
        cv.circle (rightFrame,centre2,2,(0,255,0), -1)
        cv.circle (rightFrame,centre3,2,(0,255,0), -1)
        cv.putText(rightFrame,"P",centre1, cv.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255))
        cv.putText(rightFrame,"R",centre2, cv.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255))
        cv.putText(rightFrame,"Q",centre3, cv.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255))
        if (centre4R[0] <= frameSize[0] and centre4R[1] <=frameSize[1]): 
            cv.circle (rightFrame,centre4R,3,(255,255,255), -1)
            cv.putText(rightFrame,"Z",centre4R, cv.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255))
            if test == 2:
                cv.line(rightFrame,projectedReferenceR[0][0],centre4R,(0,255,0),3)
        centre1,centre2,centre3,centre4= trackingFunctions.projectPoints(points3D,rotation,translation,cameraMatrixL,distortionL,Z_val)
        cv.circle (leftFrame,centre1,2,(0,255,0), -1)
        cv.circle (leftFrame,centre2,2,(0,255,0), -1)
        cv.circle (leftFrame,centre3,2,(0,255,0), -1)
        cv.putText(leftFrame,"P",centre1, cv.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255))
        cv.putText(leftFrame,"R",centre2, cv.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255))
        cv.putText(leftFrame,"Q",centre3, cv.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255))
        if (centre4[0] <= frameSize[0] and centre4[1] <=frameSize[1]): 
            cv.circle (leftFrame,centre4,3,(255,255,255), -1)
            cv.putText(leftFrame,"Z",centre4, cv.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255))
            if test ==2:
                cv.line(leftFrame,projectedReferenceL[0][0],centre4,(0,255,0),3)
        if test == 1:
        ##########test 1##########    
            startGraph,allDistPQ,allDistRQ,allDistPR = trackingFunctions.distanceBetweenMarkersTest(points3D,distances,startGraph,k,allDistPQ,allDistPR,allDistRQ,rightMarkers,leftMarkers,rightFrame,leftFrame)
            cv.line(rightFrame,rightMarkers[2],centre4R,(125,69,210))
        ##########test 1##########
        if test == 2:
        ##########test 2##########
            referencePoint, dist[0] =trackingFunctions.distanceBetweenPointZ(points3D,referencePoint,k,Z_val)
        ##########test 2##########
        if test == 3:
        ##########test 3##########
           # if timeCounter % 3 == 0:
            currTime = time.perf_counter()
            speed,prevTime,prevpoints3D = trackingFunctions.findSpeedofWand(points3D,prevpoints3D,currTime,prevTime)
            print ('The average wand speed is (mm/s):\t'+str(speed))
            xxValues.append(timeCounter)
            speedArray.append(speed)
                    
        ##########test 3##########
        ##########test 4##########
        if test == 4:
            points3D = points3D.transpose()
            allXp.append(points3D[0][0])
            allYp.append(points3D[0][1])
            allZp.append(points3D[0][2])
            allXq.append(points3D[2][0])
            allYq.append(points3D[2][1])
            allZq.append(points3D[2][2])
            allXr.append(points3D[1][0])
            allYr.append(points3D[1][1])
            allZr.append(points3D[1][2])
        ##########test 4##########
        if test == 5:
            points3D = points3D.transpose()
            M = trackingFunctions.findPoint(points3D[0],points3D[1],points3D[2],cv.getTrackbarPos("Z_distance","Z"))
            z_fluctuation.append(M)
        ##########test 4##########


    if test == 2:
        cv.circle (rightFrame,projectedReferenceR[0][0],3,(255,0,0), -1)
        cv.circle (leftFrame,projectedReferenceL[0][0],3,(255,0,0), -1)
        cv.putText(rightFrame,"Ref",projectedReferenceR[0][0], cv.FONT_HERSHEY_COMPLEX_SMALL,1,(255,0,0))
        cv.putText(leftFrame,"Ref",projectedReferenceL[0][0], cv.FONT_HERSHEY_COMPLEX_SMALL,1,(255,0,0))

    cv.imshow("Right Mask", rightMask)
    cv.imshow("Left Mask", leftMask)
    cv.imshow("Right Camera", rightFrame)
    cv.imshow("Left Camera", leftFrame)
    
    if (k == 27):
        break
    elif k == ord ('s'):
        cv.imwrite('Photos\FrameShots\Right_projected.png', rightFrame)
        cv.imwrite('Photos\FrameShots\Left_projected.png', leftFrame)
        cv.imwrite('Photos\FrameShots\mask1.png', mask1)
        cv.imwrite('Photos\FrameShots\mask2.png', mask2)
        cv.imwrite('Photos\FrameShots\mask3.png', mask3)
        cv.imwrite('Photos\FrameShots\mask4.png', mask4)
        cv.imwrite('Photos\FrameShots\mask5.png', mask5)
        
        np.savetxt("Distances_between_markers.txt",distances)
        if test == 2:
            f=open('z_pointDistances.txt','a')
            np.savetxt(f, dist)
            f.close()
        print ("Images have been saved")

cv.destroyAllWindows()
if test == 1:
    plt.figure(1)
    xx = np.arange(start =0, stop=len(allDistPQ))
    xx = np.linspace(start =0, stop = 10,num=len(allDistPQ))
    plt.plot(xx,allDistPQ,color = 'orange', label = 'Measured PQ Distance')
    # plt.axhline(y=max(allDistPQ),color = 'orange' ,linestyle=':')
    plt.axhline(y=np.mean(allDistPQ),color = 'orange' ,linestyle=':')
    plt.text(10,np.mean(allDistPQ)-1,str(round(np.mean(allDistPQ),2)))
    # plt.axhline(y=min(allDistPQ), color = 'orange',linestyle=':')
    plt.axhline(y=69.1,color = 'orange',linestyle='--',label= "Actual PQ distance")

    plt.plot(xx,allDistPR, color = 'green',label = 'Measured PR Distance')
    plt.axhline(y=np.mean(allDistPR),color = 'green' ,linestyle=':')
    plt.text(10,np.mean(allDistPR),str(round(np.mean(allDistPR),2)))
    # plt.axhline(y=max(allDistPR),color = 'green', linestyle=':')
    # plt.axhline(y=min(allDistPR),color = 'green', linestyle=':')
    plt.axhline(y=97,color = 'green',linestyle='--',label= "Actual PR distance")

    plt.plot(xx,allDistRQ,color = 'blue', label = 'Measured RQ Distance')
    plt.axhline(y=np.mean(allDistRQ),color = 'blue' ,linestyle=':')
    plt.text(10,np.mean(allDistRQ),str(round(np.mean(allDistRQ),2)))
    # plt.axhline(y=max(allDistRQ),color = 'blue', linestyle=':')
    # plt.axhline(y=min(allDistRQ),color = 'blue', linestyle=':')
    plt.axhline(y=69, color = 'blue',linestyle='--',label= "Actual RQ distance")

    plt.xlabel('Time (seconds)')
    plt.ylabel('Distance (mm)')
    plt.title('Measured Distances vs Actual Distances')
    plt.legend()
    plt.show()
if test == 3:
        plt.figure(1)
        plt.plot(xxValues,speedArray)
        plt.xlabel('Points')
        plt.ylabel('Speed (mm/s)')
        plt.title('Measured Average Speed of wand')
        plt.figure(2)
        w = savgol_filter(speedArray, 101, 1)
        plt.plot(xxValues, w, 'b')  # high frequency noise removed
        plt.xlabel('Points')
        plt.ylabel('Speed (mm/s)')
        plt.title('Measured Average Speed of wand')
        plt.figure(3)
        n = 100  # the larger n is, the smoother curve will be
        b = [1.0 / n] * n
        a = 1
        yy = lfilter(b,a,speedArray)
        plt.xlabel('Points')
        plt.ylabel('Speed (mm/s)')
        plt.title('Measured Average Speed of wand')
        plt.plot(xxValues, yy)  # high frequency noise removed


        plt.tight_layout()


        plt.show()
if test == 4:
    xx = np.arange(start =0, stop=len(allXp))
    allXp=np.array(allXp)
    allYp=np.array(allYp)
    allZp=np.array(allZp)
    allXq=np.array(allXq)
    allYq=np.array(allYq)
    allZq=np.array(allZq)
    allXr=np.array(allXr)
    allYr=np.array(allYr)
    allZr=np.array(allZr)
    plt.figure("X co-ordinates") 
    plt.plot(xx,allXp,label = "Point P")
    plt.plot(xx,allXq,label = "Point Q")
    plt.plot(xx,allXr,label = "Point R")
    plt.xlabel('Points')
    plt.ylabel('X Distance (mm)')
    plt.title('X co-ordinates variance')
    plt.legend()
    plt.tight_layout()
    plt.figure("Y co-ordinates") 
    plt.plot(xx,allYp,label = "Point P")
    plt.plot(xx,allYq,label = "Point Q")
    plt.plot(xx,allYr,label = "Point R")
    plt.xlabel('Points')
    plt.ylabel('Y Distance (mm)')
    plt.title('Y co-ordinates variance')
    plt.legend()
    plt.tight_layout()
    plt.figure("Z co-ordinates") 
    plt.plot(xx,allZp,label = "Point P")
    plt.plot(xx,allZq,label = "Point Q")
    plt.plot(xx,allZr,label = "Point R")

    plt.tight_layout()  
    xp_var = np.var(allXp)
    yp_var = np.var(allYp)
    zp_var = np.var(allZp)
    xq_var = np.var(allXq)
    yq_var = np.var(allYq)
    zq_var = np.var(allZq)
    xr_var = np.var(allXr)
    yr_var = np.var(allYr)
    zr_var = np.var(allZr)
    print ("Xp "+str(xp_var))
    print ("Yp "+str(yp_var))
    print ("Zp "+str(zp_var))
    print ("Xq "+str(xq_var))
    print ("Yq "+str(yq_var))
    print ("Zq "+str(zq_var))
    print ("Xr "+str(xr_var))
    print ("Yr "+str(yr_var))
    print ("Zr "+str(zr_var))
    plt.show()  
if test == 5:
    xx = np.arange(start =0, stop=len(z_fluctuation))
    plt.figure(figsize=(12,6))
    z_fluctuation = np.array(z_fluctuation)
    plt.plot (xx,z_fluctuation[:,0],label = "X coordinate")
    plt.plot (xx,z_fluctuation[:,1],label = "Y coordinate")
    plt.plot (xx,z_fluctuation[:,2],label = "Z coordinate")
    plt.xlabel('Points')
    plt.ylabel('Coordinate (mm)')
    plt.title('Z stability') 
    print ("X "+str(np.var(z_fluctuation[:,0])))
    print ("Y "+str(np.var(z_fluctuation[:,1])))
    print ("Z "+str(np.var(z_fluctuation[:,2])))
    errorX = max(z_fluctuation[:,0])-min(z_fluctuation[:,0])
    errorY = max(z_fluctuation[:,1])-min(z_fluctuation[:,1])
    errorZ = max(z_fluctuation[:,2])-min(z_fluctuation[:,2])
    print ("Error" + str(np.sqrt(np.var(z_fluctuation[:,0])+np.var(z_fluctuation[:,1])+np.var(z_fluctuation[:,2]))))
    plt.legend()   
    plt.tight_layout()
    plt.show() 


rightCap.release()
leftCap.release()