from picamera import PiCamera
from time import sleep
import numpy as np
import cv2 as cv
from Chessboard_detection import pi_debugging

camera = PiCamera()
camera.resolution = (480, 640)
camera.rotation = 270

pathManager = pi_debugging.imgDBManager()

camera.start_preview()
sleep(5)
output = np.empty((640, 480, 3), dtype=np.uint8)
camera.capture(output, 'rgb')
pathManager.saveDBImage(output)
camera.stop_preview()

camera.close()