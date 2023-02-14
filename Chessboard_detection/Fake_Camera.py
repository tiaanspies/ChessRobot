import os
import cv2 as cv
from time import sleep
# for PhoneCamera:
import requests
import numpy as np
from Chessboard_detection import Chess_Vision
from picamera import PiCamera

# import Chess_Vision

#File seperator depending on OS \\ for Windows
# / for raspberry pi

SEP = "/"

class FakeCamera:
    def __init__(self, res, absPath) -> None:

        self.path_full = absPath

        self.cameraRes = res

        self.stateNum = -1
    
    def read(self):
        if self.stateNum <0:
            self.frame = cv.imread(self.path_full + SEP + "empty.JPG")
            self.frame = cv.resize(self.frame, self.cameraRes)
        elif self.stateNum >= 0:
            self.frame = cv.imread(self.path_full + SEP +str(self.stateNum) +".JPG")
            if self.frame is not None:
                self.frame = cv.resize(self.frame, self.cameraRes)

        self.stateNum += 1

        ret = self.frame is not None

        return ret, self.frame

    def isOpened(self):
        res = cv.imread(self.path_full + SEP + "empty.JPG")

        return res is not None

    def release(self):
        return True

class PhoneCamera:
    def __init__(self, res, absPath) -> None:

        self.cameraRes = res
        self.absPath = absPath

        self.stateNum = -1

        self.url = "http://10.0.0.220:8080//shot.jpg" # url from Noah's house
        # self.url = "http://10.192.48.63:8080///shot.jpg" # update this with the url from web
    
    def read(self):
        # self.stateNum = 10 # use this line to skip the saved empty picture and do it by hand
        if self.stateNum <= 0:
            self.path_full = self.absPath
            self.frame = cv.imread(self.path_full + "\\empty.JPG")
            self.frame = cv.resize(self.frame, self.cameraRes)
        elif self.stateNum > 0:
            img_resp = requests.get(self.url)
            img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
            self.frame = cv.imdecode(img_arr, -1)
            self.frame = cv.resize(self.frame, self.cameraRes)
        ret = self.frame is not None

        self.stateNum += 1
        return ret, self.frame

    def isOpened(self):
        ret, _ = self.read()

        return ret

    def release(self):
        return True


class RPiCamera:

    def __init__(self, res, absPath) -> None:

        self.camera = PiCamera()
        self.cameraRes = res
        self.camera.resolution = res
        self.camera.rotation = 270

        self.path_full = absPath
        self.stateNum = -1
    
    def read(self):
        # self.stateNum = 10 # use this line to skip the saved empty picture and do it by hand
        if self.stateNum <= 0:
            self.frame = cv.imread(self.path_full + SEP +"empty.jpg")

            if self.frame is None:
                print("Cannot read stored initialization file")
                exit()

            self.frame = cv.resize(self.frame, self.cameraRes)
        elif self.stateNum > 0:
            self.camera.start_preview()
            sleep(5)
            output = np.empty((self.cameraRes[1], self.cameraRes[0], 3), dtype=np.uint8)
            self.camera.capture(output, 'rgb')
            self.camera.stop_preview()
            # self.camera.close()
            self.frame = output.copy()
            print(self.frame.shape)
        ret = self.frame is not None

        self.stateNum += 1

        cv.imwrite(self.path_full + SEP +"initImg.jpg", self.frame)
        return ret, self.frame

    def isOpened(self):
        ret, _ = self.read()

        return ret

    def release(self):
        return True
'''
# Replace the below URL with your own. Make sure to add "/shot.jpg" at last.
url = "http://192.168.0.103:8080/shot.jpg"

# While loop to continuously fetching data from the Url
while True:
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv.imdecode(img_arr, -1)
    img = imutils.resize(img, width=1000, height=1800)
    cv.imshow("Android_cam", img)
  
    # Press Esc key to exit
    if cv.waitKey(1) == 27:
        break
  
cv.destroyAllWindows()
'''