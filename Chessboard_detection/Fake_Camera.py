import os
import cv2 as cv

class FakeCamera:
    def __init__(self, res) -> None:
        # "Chessboard_detection\Test_Images\IMG_0165.png"
        # C:\Users\spies\OneDrive\Documents\Chess Robot\ChessRobot\Chessboard_detection\Test_Images\b_w_game\empty.JPG
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.path_full = dir_path + "\Test_Images\\b_w_game\\"
        self.cameraRes = res

        self.stateNum = -1
    
    def read(self):
        if self.stateNum == -1:
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