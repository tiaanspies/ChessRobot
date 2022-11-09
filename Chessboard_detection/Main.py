import numpy as np
import cv2 as cv

BOARD_SIZE = (8, 8)
BOARD_SIZE_INT = (7, 7)

CAMERA_RESOLUTION = (480, 640)

def empty(a):
    pass

class ChessBoard:
    def __init__(self) -> None:
        self.currentConfig = np.zeros(BOARD_SIZE, dtype=bool)  
        self.cornersInterior = np.zeros(BOARD_SIZE_INT, dtype=np.uint32)
        self.cornersAll = np.zeros(BOARD_SIZE, dtype=np.uint32)
        self.initialImage = np.zeros(CAMERA_RESOLUTION)
    
    def setInitialImage(self, camera):
        while True:
            ret, frame = camera.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                exit()
            
            cv.imshow('Initial Image', frame)

            if cv.waitKey(1) == ord('q'):
                resp = input("Are you happy with this initialization image? (Y/N)")
            
                if resp == 'Y':
                    self.initialImage = frame
                    cv.destroyWindow('Initial Image')
                    break

    def threshHoldAndFindBoard(self, img, threshold, erodeSize):
        # Threshold
        ret, thresh = cv.threshold(img, threshold, 255, cv.THRESH_BINARY_INV)
        
        # If erodeSize is invalid skip eroding and dilating
        if erodeSize > 0:
            element = cv.getStructuringElement(cv.MORPH_RECT, (erodeSize, erodeSize))
            erode = cv.erode(thresh, element)
            dilate = cv.dilate(erode, element)
        else:
            dilate = thresh

        didFind, intCorners = cv.findChessboardCorners(dilate, BOARD_SIZE_INT)

        return didFind, intCorners

    def findOptimalThreshold(self, img, blurSize=3, erodeSize=3):
        threshold = 127

        # Convert img to grayscale
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # If blursize is invalid then skip blurring
        if blurSize > 0:
            blur = cv.blur(gray, (blurSize, blurSize))
        else:
            blur = gray

        didFind, intCorners = self.threshHoldAndFindBoard(blur, threshold, erodeSize)
        cv.drawChessboardCorners(img, BOARD_SIZE_INT, intCorners, didFind)

        return img


def main():
    cam = cv.VideoCapture(0)
    if not cam.isOpened():
        print("Cannot open camera")
        exit()

    board = ChessBoard()
    board.setInitialImage(cam)
    x = board.findOptimalThreshold(board.initialImage)
    cv.imshow('Detect', x)

    while cv.waitKey(1) != ord('q'):
        pass

    cam.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()