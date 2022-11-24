import os
import cv2 as cv

class FakeCamera:
    def __init__(self, res, absPath) -> None:

        self.path_full = absPath

        self.cameraRes = res

        self.stateNum = -1
    
    def read(self):
        if self.stateNum <0:
            self.frame = cv.imread(self.path_full + "\\empty.JPG")
            self.frame = cv.resize(self.frame, self.cameraRes)
        elif self.stateNum >= 0:
            self.frame = cv.imread(self.path_full + "\\"+str(self.stateNum) +".JPG")
            if self.frame is not None:
                self.frame = cv.resize(self.frame, self.cameraRes)

        self.stateNum += 1

        ret = self.frame is not None

        return ret, self.frame

    def isOpened(self):
        res = cv.imread(self.path_full + "\\empty.JPG")

        return res is not None

    def release(self):
        return True