import os
import cv2 as cv

class FakeCamera:
    def __init__(self, image_name, res) -> None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path_full = dir_path + "\\" + image_name
        self.frame = cv.imread(image_name)

        self.frame = cv.resize(self.frame, res)
    
    def read(self):
        ret = self.frame is not None

        return ret, self.frame

    def isOpened(self):
        return self.frame is not None