import os, requests, io
import cv2 as cv
from time import sleep
import logging
import numpy as np
from pathlib import Path
try:
    from Chessboard_detection import pi_debugging
except ModuleNotFoundError:
    import pi_debugging

try:
    from picamera import PiCamera
except ModuleNotFoundError:
    print("Can only use RPICam when on raspberry pi")

class FakeCamera:
    def __init__(self, res, absPath, startNum = 0) -> None:

        self.path_full = absPath
        self.cameraRes = res
        self.stateNum = startNum

    def readCalibMatrix(self):
        """
        Return the calibration matrix that was saved to cameraProperties.out
        """
        camera_calib_file = Path("Camera", "cameraProperties.out")

        with open(str(camera_calib_file.resolve()), 'rb') as f:
            camera_calib = np.load(f, allow_pickle=True)

        camera_matrix = camera_calib[0]
        dist_matrix = camera_calib[1]

        return camera_matrix, dist_matrix
    
    def read(self):
        if self.stateNum <0:
            self.frame = cv.imread(os.path.join(self.path_full, "empty.JPG"))
            self.frame = cv.cvtColor(self.frame, cv.COLOR_RGB2BGR)
            self.frame = cv.resize(self.frame, self.cameraRes)
        elif self.stateNum >= 0:
            self.frame = cv.imread(os.path.join(self.path_full, str(self.stateNum)+".JPG"))
            # self.frame = cv.cvtColor(self.frame, cv.COLOR_RGB2BGR)
            if self.frame is not None:
                self.frame = cv.resize(self.frame, self.cameraRes)

        self.stateNum += 1

        ret = self.frame is not None

        return ret, self.frame

    def isOpened(self):
        res = cv.imread(os.path.join(self.path_full, "empty.JPG"))

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
            self.frame = cv.imread(os.path.join(self.path_full, "empty.JPG"))
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
    def __init__(self, absPath=None,res=(480, 640), storeImgHist=True, loadSavedFirst=True) -> None:
        self.imgHistDB = pi_debugging.imgDBManager()

        self.camera = PiCamera()
        self.cameraRes = res
        self.camera.resolution = res
        self.camera.rotation = 270

        self.path_full = absPath
        self.storeImgHist = storeImgHist

        self.camera_matrix, self.dist_matrix = self.readCalibMatrix()

        # Change statenum to -1 to use saved picture as first picture.
        # Change statenum to 1 to used camera for all pictures
        print(
            f"Creating camera object.\n"\
            f"\t loadSavedFirst: {loadSavedFirst}\n"\
            f"\t storeImgHist: {storeImgHist}\n"\
            f"\t camera.rotation: {self.camera.rotation}"
        )

        if loadSavedFirst:
            if absPath is None:
                raise ValueError("absPath must be given if loadSavedFirst is true. "\
                                  "Set loadSavedFirst to false to not load stored image first")
            self.stateNum = -1
        else:
            self.stateNum = 1

        sleep(2)

    def __del__(self):
        self.camera.close()

    def readCalibMatrix(self):
        """
        Return the calibration matrix that was saved to cameraProperties.out
        """
        camera_calib_file = Path("Camera", "cameraProperties rpi.out")

        with open(str(camera_calib_file.resolve()), 'rb') as f:
            camera_calib = np.load(f, allow_pickle=True)

        camera_matrix = camera_calib[0]
        dist_matrix = camera_calib[1]

        return camera_matrix, dist_matrix
    
    def read(self):
        
        if self.stateNum <= 0:
            self.frame = cv.imread(os.path.join(self.path_full, "empty.jpg"))

            if self.frame is None:
                print("Cannot read stored initialization file")
                exit()

            self.frame = cv.resize(self.frame, self.cameraRes)
        elif self.stateNum > 0:
            # Read Stream Method
            output = np.empty((self.cameraRes[1], self.cameraRes[0], 3), dtype=np.uint8)
            self.camera.capture(output, 'rgb', use_video_port=True)
            
            self.frame = output.copy()
            self.frame = cv.cvtColor(self.frame, cv.COLOR_RGB2BGR)

        ret = self.frame is not None

        self.stateNum += 1

        if ret:
            self.frame = cv.undistort(self.frame, self.camera_matrix, self.dist_matrix)

        if self.storeImgHist:
            self.imgHistDB.saveDBImage(self.frame)

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
def read_images_from_webcam():
    # Create a VideoCapture object to access the webcam (0 represents the default camera)
    cap = cv.VideoCapture(0)

    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()

        # Check if the frame was captured successfully
        if not ret:
            print("Failed to capture frame.")
            break

        # Display the frame in a window
        cv.imshow("Webcam", frame)

        # Exit the loop if the 'q' key is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    cap.release()
    cv.destroyAllWindows()

class LaptopCamera:
    def __init__(self, camera_num=0) -> None:
        self.cam = cv.VideoCapture(camera_num)
        self.camera_matrix, self.dist_matrix = self.readCalibMatrix()

    def readCalibMatrix(self):
        """
        Return the calibration matrix that was saved to cameraProperties.out
        """
        camera_calib_file = Path("Camera", "laptopCalib.npy")

        with open(str(camera_calib_file.resolve()), 'rb') as f:
            camera_calib = np.load(f, allow_pickle=True)

        camera_matrix = camera_calib[0]
        dist_matrix = camera_calib[1]

        return camera_matrix, dist_matrix
    
    def read(self):
        ret, frame = self.cam.read()
        return ret, frame

    def isOpened(self):
        res, _ = self.cam.read()

        return res is not None

    def release(self):
        return True