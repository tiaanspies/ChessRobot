import cv2 as cv
import numpy as np
import time

def prepareMatching(filePath):
    marker = cv.imread (filePath,0)
    ret, thresh = cv.threshold(marker, 127, 255,0)
    contours,hierarchy = cv.findContours(thresh,2,1)
    cnt1 = contours [0]
    return cnt1

def empty(x):
    pass


def createTrackbars(camera):
    cv.namedWindow (camera)
    #print (camera)
    cv.createTrackbar("Hue Min", camera,0,360,empty)
    cv.setTrackbarPos("Hue Min", camera, 0)

    cv.createTrackbar("Hue Max", camera,360,360,empty)
    cv.setTrackbarPos("Hue Max", camera, 360)

    cv.createTrackbar("Sat Min",camera,0,255,empty)
    cv.setTrackbarPos("Sat Min", camera, 0)

    cv.createTrackbar("Sat Max",camera,255,255,empty)
    cv.setTrackbarPos("Sat Max", camera, 255)

    cv.createTrackbar("Val Min",camera,0,255,empty)
    cv.setTrackbarPos("Val Min", camera, 0)

    cv.createTrackbar("Val Max",camera,255,255,empty)
    cv.setTrackbarPos("Val Max", camera, 255)

def threshTrackBar(camera):
    cv.namedWindow (camera)
    cv.createTrackbar("threshVal",camera, 0,255,empty)
    cv.setTrackbarPos("threshVal",camera, 0)

def getRangeHSV(camera):
    h_min = int(cv.getTrackbarPos ("Hue Min",camera))
    h_max = int(cv.getTrackbarPos ("Hue Max",camera))
    S_min = int(cv.getTrackbarPos ("Sat Min",camera))
    S_max = int(cv.getTrackbarPos ("Sat Max",camera))
    V_min = int(cv.getTrackbarPos ("Val Min",camera))
    V_max = int(cv.getTrackbarPos ("Val Max",camera))
    return ((h_min,S_min,V_min),(h_max,S_max,V_max))


def createMask(frame,HSVrange):
    mask = cv.medianBlur(frame,3)
    mask = cv.cvtColor(mask, cv.COLOR_BGR2HSV)
    mask = cv.inRange(mask, HSVrange[0], HSVrange[1])
    mask = cv.erode(mask, None, iterations=5)
    mask = cv.dilate(mask, None, iterations=5)
    return mask

def thresholdMask(frame, threshVal):
    mask = cv.GaussianBlur(frame,(11,11),0)
    #mask = cv.bilateralFilter(frame,9,75,75)
    mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    retval, threshold = cv.threshold(mask, threshVal, 255, cv.THRESH_BINARY)
    # cv.imshow('original',frame)
    # cv.imshow('threshold',threshold)
    mask = cv.erode(threshold, None, iterations=4)
    mask = cv.dilate(mask, None, iterations=4)
    return mask


def findMarkers(mask,shape):
    contours,hierachies = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    center = None
    contour_list =[]
    coord = []
    if len(contours) > 0:
        for contour in contours:
            ret = cv.matchShapes(shape,contour,1,0.0)
            if (ret < 0.5):
                contour_list.append(contour)

        for currentCircle in contour_list:
            ((x,y),radius) = cv.minEnclosingCircle (currentCircle)
            M = cv.moments(currentCircle)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            coord.append(center)
    return coord 

def drawMarkers(points,frame):
    for center in points:
        cv.circle(frame, center, 5, (255,0,0), -1)

def findPoint(P,R,Q,d2):
    X_PR= R[0]-P[0] 
    Y_PR = R[1]-P[1]
    Z_PR = R[2]-P[2]
    PQ = Q-P
    PR = R-P
    n = np.cross(PR,PQ)
    a,b,c = n 
    d = np.dot(n, P)
    #d2 = 40#48 #distance from Z to Q 
    d1 = d2**2+69.1**2-2*(d2)*(69.1)*(1/np.sqrt(2)) #77.5#49 #distance from Z to P/R
    
    A = np.array([[a,b,c],[X_PR,Y_PR,Z_PR],[2*Q[0]-2*P[0],2*Q[1]-2*P[1],2*Q[2]-2*P[2]]])

    B = np.array([[d],[X_PR*Q[0]+Y_PR*Q[1]+Z_PR*Q[2]],[Q[0]**2-P[0]**2+Q[1]**2-P[1]**2+Q[2]**2-P[2]**2+d1-d2**2]])
    x = np.linalg.lstsq(A,B,rcond=None)[0]
    x = np.reshape(x,3)
    return x

def get3Dpoints(rightMarkers,leftMarkers,fundamental,projMatrix_R,projMatrix_L):
    rightPoints = np.array(rightMarkers)
    leftPoints = np.array(leftMarkers)
    rightPoints = rightPoints.astype(float)
    leftPoints = leftPoints.astype(float)
    LEFTpoints = []
    RIGHTpoints = []
    for left,right in zip(leftPoints,rightPoints):
        LEFTpoints.append([left])
        RIGHTpoints.append([right])
    LEFTpoints = np.array(LEFTpoints)
    RIGHTpoints = np.array(RIGHTpoints)
    if ((len(rightPoints)==3 and len(leftPoints)==3) or (len(rightPoints)==1 and len(leftPoints)==1)) and len(rightPoints) == len(leftPoints):
        LEFTpoints = np.reshape(leftPoints,(1,3,2))
        RIGHTpoints = np.reshape(rightPoints,(1,3,2))
        RIGHTpoints,LEFTpoints = cv.correctMatches(fundamental,RIGHTpoints,LEFTpoints)
        points4D=cv.triangulatePoints(projMatrix_R,projMatrix_L,RIGHTpoints,LEFTpoints) ## check correct Matches
        points3D = points4D/points4D[-1]
        points3D = points3D[ :-1]
        return points3D

def projectPoints(points3D,R,T,cameraMatrix,distortion,d2):
    points3D = points3D.transpose()
    M = findPoint(points3D[0],points3D[1],points3D[2],d2)
    reprojected2D,jacobian_points= cv.projectPoints(points3D,R,T,cameraMatrix, distortion)
    projected,jacobian= cv.projectPoints(M,R,T,cameraMatrix, distortion)
    centre1=(int(reprojected2D[0][0][0]),int(reprojected2D[0][0][1]))
    centre2=(int(reprojected2D[1][0][0]),int(reprojected2D[1][0][1]))
    centre3=(int(reprojected2D[2][0][0]),int(reprojected2D[2][0][1]))
    centre4=(int(projected[0][0][0]),int(projected[0][0][1]))
    return centre1,centre2,centre3,centre4

def distanceBetweenMarkersTest(points3D,distances,startGraph,k,allDistPQ,allDistPR,allDistRQ,rightMarkers,leftMarkers,rightFrame,leftFrame):
    points3D = points3D.transpose()
    dPR = np.linalg.norm(points3D[0] - points3D[1])  
    dRQ = np.linalg.norm(points3D[1] - points3D[2])                  
    dPQ = np.linalg.norm(points3D[0] - points3D[2])  
    distances[0] =dPR
    distances[1] =dPQ
    distances [2] =dRQ 
    cv.line (rightFrame,rightMarkers[0],rightMarkers[1],(0,0,255)) #PR
    centrePR = ((rightMarkers[0][0]+rightMarkers[1][0])//2,(rightMarkers[0][1]+rightMarkers[1][1])//2+25)
    cv.putText(rightFrame,"PR-"+str(round(dPR,2)),centrePR, cv.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255))

    cv.line (rightFrame,rightMarkers[1],rightMarkers[2],(0,255,255)) #RQ
    centreRQ = ((rightMarkers[1][0]+rightMarkers[2][0])//2-50,(rightMarkers[1][1]+rightMarkers[2][1])//2)
    cv.putText(rightFrame,"RQ-"+str(round(dRQ,2)),centreRQ, cv.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,255))

    cv.line (rightFrame,rightMarkers[0],rightMarkers[2],(255,0,255)) #PQ
    centrePQ = ((rightMarkers[0][0]+rightMarkers[2][0])//2+20,(rightMarkers[0][1]+rightMarkers[2][1])//2)
    cv.putText(rightFrame,"PQ-"+str(round(dPQ,2)),centrePQ, cv.FONT_HERSHEY_COMPLEX_SMALL,1,(255,0,255))


    cv.line (leftFrame,leftMarkers[0],leftMarkers[1],(0,0,255)) #PR
    cv.line (leftFrame,leftMarkers[1],leftMarkers[2],(0,255,255)) #RQ
    cv.line (leftFrame,leftMarkers[0],leftMarkers[2],(255,0,255)) #PQ

    if startGraph ==1:               
        allDistPQ.append(dPQ)
        allDistPR.append(dPR)
        allDistRQ.append(dRQ)
    print ("PR:\t"+str(dPR)+"\tPQ:\t"+str(dPQ)+"\tRQ:\t"+str(dRQ))
    if k == ord ('q'):
        if startGraph ==0:
            startGraph = 1
            print ("Starting graphing")
        else:
            startGraph = 0
            print ("Ending graphing")
    return startGraph,allDistPQ,allDistRQ,allDistPR            



def distanceBetweenPointZ(points3D,referencePoint,k,d2):
    points3D = points3D.transpose()
    pointZ = findPoint(points3D[0],points3D[1],points3D[2],d2)
    if k == ord ('q'):
        referencePoint = pointZ
    dist = np.linalg.norm(referencePoint - pointZ)
    print ("Distance to Ref point:\t"+str(dist))  

    return referencePoint,dist        

def findSpeedofWand(points3D,prevpoints3D,currTime,prevTime):
    dP = np.linalg.norm(points3D[0] - prevpoints3D[0]) 
    dR = np.linalg.norm(points3D[1] - prevpoints3D[1]) 
    dQ = np.linalg.norm(points3D[2] - prevpoints3D[2]) 
    d = cv.mean([dP,dR, dQ])
    speed = d/(currTime-prevTime)
    prevTime = currTime
    prevpoints3D = points3D
    return speed,prevTime,prevpoints3D

    







