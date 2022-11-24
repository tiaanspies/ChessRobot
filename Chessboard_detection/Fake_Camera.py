import os
import cv2 as cv

class FakeCamera:
    def __init__(self, res, relPath) -> None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        # self.path_full = dir_path + "\Test_Images\\b_w_game\\"
        self.path_full = dir_path + relPath
        # self.path_full = dir_path + "\Test_Images\\brown_pieces\\"
        self.cameraRes = res

        self.stateNum = -1
    
    def read(self):
        if self.stateNum <0:
            self.frame = cv.imread(self.path_full + "empty.JPG")
            self.frame = cv.resize(self.frame, self.cameraRes)
        elif self.stateNum >= 0:
            self.frame = cv.imread(self.path_full + str(self.stateNum) +".JPG")
            self.frame = cv.resize(self.frame, self.cameraRes)

        self.stateNum += 1

        ret = self.frame is not None

        return ret, self.frame

    def isOpened(self):
        return True

    def release(self):
        return True