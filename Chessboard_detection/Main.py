import numpy as np
import cv2 as cv
import math
from pathlib import Path
from matplotlib import pyplot as plt

import Fake_Camera



CAMERA_RESOLUTION = (640, 480)

def empty(a):
    pass

def minPos(arr):
    found = False
    index = 0
    for i, item in enumerate(arr):
        if item:
            index = i
            found = True
            break

    return found, index

def maxPos(arr):
    found = False
    index = 0
    for i, item in reversed(list(enumerate(arr))):
        if item:
            index = abs(i)
            found = True
            break

    return found, index


class ChessBoard:
    def __init__(self, camera) -> None:
        self.initialImage = np.zeros(CAMERA_RESOLUTION)
        self.camera = camera

        self.BOARD_SIZE = (9, 9)
        self.BOARD_SIZE_INT = (7, 7)

        self.setInitialImage(camera)

        self.thresholdOpt = self.findOptimalThreshold(self.initialImage)

        s, cornersInt  = self.findBoardCorners(self.initialImage, self.thresholdOpt)

        cornersIntReshaped = np.reshape(cornersInt, self.BOARD_SIZE_INT + (2,))
        cornersIntReoriented = self.makeTopRowFirst(cornersIntReshaped)

        self.cornersExt = self.estimateExternalCorners(cornersIntReoriented)

    
    def setInitialImage(self, camera, confirmFirst=False):
        while True:
            #Get image frame
            ret, frame = camera.read()

            if not ret:
                print("Can't receive frame (stream end?).")
                raise("Could not receive frame from camera to set initial image")
            
            # Display video
            if confirmFirst:
                cv.imshow('Initial Image', frame)

                # Pause video when q is pressed
                if cv.waitKey(1) == ord('q'):
                    resp = input("Are you happy with this initialization image? (y/n)")
                
                    # Check if the image is good enough
                    if resp == 'y':
                        cv.destroyWindow('Initial Image')
                        break

            if not confirmFirst:
                break
        
        self.initialImage = frame.copy()
        

    def findGrayBlur(self, img, blurSize):
        # Convert img to grayscale
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # If blursize is invalid then skip blurring
        if blurSize > 0:
            blur = cv.blur(gray, (blurSize, blurSize))
        else:
            blur = gray

        return blur

    def threshHoldAndFindBoard(self, grayBlurImg, threshold, erodeSize):
        # Threshold
        ret, thresh = cv.threshold(grayBlurImg, threshold, 255, cv.THRESH_BINARY_INV)
        
        # If erodeSize is invalid skip eroding and dilating
        if erodeSize > 0:
            element = cv.getStructuringElement(cv.MORPH_RECT, (erodeSize, erodeSize))
            erode = cv.erode(thresh, element)
            dilate = cv.dilate(erode, element)
        else:
            dilate = thresh

        # find chessboard corners, If its too slow add the fast stop param
        didFind, intCorners = cv.findChessboardCorners(dilate, self.BOARD_SIZE_INT)

        return didFind, intCorners

    def findBoardCorners(self, img, threshold, blursize=3, erodeSize=3):
        grayBlur = self.findGrayBlur(img, blursize)
        ret, corners = self.threshHoldAndFindBoard(grayBlur, threshold, erodeSize)
        return ret, corners

    def findOptimalThreshold(self, img, blurSize=3, erodeSize=3):
        stepSize = 10

        #Find gray blurry img
        grayBlur = self.findGrayBlur(img, blurSize)

        #do a first rough pass by checking all values with step size
        # There is no point in checking 0 and 255
        result_array = [0]*(math.ceil(255/stepSize)-2)
        for i in range(1, math.floor(255/stepSize)):
            result_array[i-1], _ = self.threshHoldAndFindBoard(grayBlur, i*stepSize, erodeSize)

        #Find min and max threshold bound
        respMin, minBound = minPos(result_array)
        minBound = stepSize + minBound*stepSize - 1
        respMax, maxBound = maxPos(result_array)
        maxBound = stepSize + maxBound*stepSize + 1

        if not respMin or not respMax:
            print("Could not find successful threshold")
            exit()

        # finetune min
        iter = 0
        result = False
        while iter < 10:
            result, _ = self.threshHoldAndFindBoard(grayBlur, minBound, erodeSize)
            if not result:
                break
            minBound -= 1
        else:
            raise("minbound never found finetuned val")

        # finetune max
        result = False
        while iter < 10:
            result, _ = self.threshHoldAndFindBoard(grayBlur, maxBound, erodeSize)
            if not result:
                break
            maxBound += 1
        else:
            raise("maxBound never found finetuned val")

        # return average of minBound and Maxbound to hopefully be most reliable threshold
        thresholdOpt = int((maxBound+minBound)/2)
        return thresholdOpt 

    def estimateExternalCorners(self, cornersInt):

        diffRow = np.diff(cornersInt, axis=0)
        meanDiffRow = np.mean(diffRow, axis=0)
        externTop = 2*cornersInt[None, 0, :, :] - cornersInt[None, 1, :, :]
        externBottom = 2*cornersInt[None, -1, :, :] - cornersInt[None, -2, :, :]

        difCol = np.diff(cornersInt, axis=1)
        meanDiffCol = np.mean(difCol, axis=1)
        meanDiffCol = np.reshape(meanDiffCol, (7, 1, 2))

        externleft = 2*cornersInt[:, None, 0, :] - cornersInt[:, None, 1, :]
        externRight = 2*cornersInt[:, None, -1, :] - cornersInt[:, None, -2, :]

        topLeft = 3*cornersInt[None, 0, None, 0, :] - cornersInt[None, 1, None, 0, :] - cornersInt[None, 0, None, 1, :]
        topRight = 3*cornersInt[None, 0, None, -1, :] - cornersInt[None, 1, None, -1, :] - cornersInt[None, 0, None, -2, :]
        botLeft = 3*cornersInt[None, -1, None, 0, :] - cornersInt[None, -2, None, 0, :] - cornersInt[None, -1, None, 1, :]
        botRight = 3*cornersInt[None, -1, None, -1, :] - cornersInt[None, -2, None, -1, :] - cornersInt[None, -1, None, -2, :]

        cornersExtRow0 = np.hstack([topLeft, externTop, topRight])
        cornersExtRow1 = np.hstack([externleft, cornersInt, externRight])
        cornersExtRow2 = np.hstack([botLeft, externBottom, botRight])

        cornersExt = np.vstack([cornersExtRow0, cornersExtRow1, cornersExtRow2])

        return cornersExt

    def makeTopRowFirst(self, corners):
        # TODO: adjust angle to keep top row at the top of image
        # The array should start form top left.
        firstPoint = corners[0][0]
        lastPointFirstRow = corners[0][-1]

        # firstRowAngle = np.arctan2()
        # newArr = np.transpose(corners, axes=(1, 0, 2))
        # newArr = np.flip(newArr, axis=1)
        newArr = corners
        return newArr

    def getCurrentPositions(self):
        # Read new image from camera object
        s, img = self.camera.read()

        #TODO: Change the detection to work with board pieces on
        # it will not work work with findBoardCorners

        cornersExt = np.vstack([
                self.cornersExt[None, 0, 0, :],
                self.cornersExt[None, 0, -1, :],
                self.cornersExt[None, -1, -1, :],
                self.cornersExt[None, -1, 0, :]
            ])

        blur = cv.GaussianBlur(img, (11,11), 0)
        mask = np.zeros(img.shape, dtype=np.uint8)
        mask = cv.fillConvexPoly(mask, np.int32(cornersExt), color=(255, 255, 255))

        masked = cv.bitwise_and(blur, mask)
        cv.drawChessboardCorners(img, self.BOARD_SIZE, cornersExt, True)
        showImg(masked)
        
        return s, cornersExt

def showImg(img):
    while cv.waitKey(1) != ord('q'):
        cv.imshow('name', img)


def main():
    # Open Video camera
    # cam = cv.VideoCapture(0)
    cam = Fake_Camera.FakeCamera(CAMERA_RESOLUTION)    

    if not cam.isOpened():
        raise("Cannot open camera.")

    # Initialize ChessBoard object and select optimal thresh
    board = ChessBoard(cam)

    # display video of chessboard with corners
    while cv.waitKey(1) != ord('q'):
        ret, img = cam.read()
        
        s, corners = board.getCurrentPositions()
        # cv.drawChessboardCorners(img, BOARD_SIZE_INT, corners, s)
        
        # cv.imshow("Image", img)

    cam.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()