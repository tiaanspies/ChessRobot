import numpy as np
import cv2 as cv
import math
from pathlib import Path

BOARD_SIZE = (8, 8)
BOARD_SIZE_INT = (7, 7)

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
    def __init__(self) -> None:
        self.currentConfig = np.zeros(BOARD_SIZE, dtype=bool)  
        self.cornersInterior = np.zeros(BOARD_SIZE_INT, dtype=np.uint32)
        self.cornersAll = np.zeros(BOARD_SIZE, dtype=np.uint32)
        self.initialImage = np.zeros(CAMERA_RESOLUTION)
        self.thresholdOpt = 0
    
    def setInitialImage(self, camera):
        while True:
            #Get image frame
            ret, frame = camera.read()
            # frame = cv.imread(r"C:\Users\spies\OneDrive\Documents\Chess Robot\ChessRobot\Chessboard_detection\Test_Images\IMG_0165.png")
            # frame = cv.resize(frame, CAMERA_RESOLUTION)
            # ret = True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                exit()
            
            # Display video
            cv.imshow('Initial Image', frame)

            # Pause video when q is pressed
            if cv.waitKey(1) == ord('q'):
                resp = input("Are you happy with this initialization image? (y/n)")
            
                # Check if the image is good enough
                if resp == 'y':
                    self.initialImage = frame
                    cv.destroyWindow('Initial Image')
                    break

        return

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
        didFind, intCorners = cv.findChessboardCorners(dilate, BOARD_SIZE_INT)

        return didFind, intCorners

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
        self.thresholdOpt = int((maxBound+minBound)/2)
        return self.thresholdOpt 


def main():
    # Open Video camera
    cam = cv.VideoCapture(0)
    if not cam.isOpened():
        print("Cannot open camera")
        exit()

    # Initialize ChessBoard object and select initialimage.
    board = ChessBoard()
    board.setInitialImage(cam)

    # Use initial image to find optimal threshold
    opt = board.findOptimalThreshold(board.initialImage)
    print(opt)

    while cv.waitKey(1) != ord('q'):
        pass

    cam.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()