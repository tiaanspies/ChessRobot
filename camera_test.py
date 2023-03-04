import os
from Chessboard_detection import pi_debugging, Camera_Manager

# start camera
dirPath = os.path.dirname(os.path.realpath(__file__))
relPath = "/Chessboard_detection/TestImages/Temp"
cam = Camera_Manager.RPiCamera(res=(480, 640), absPath=dirPath+relPath)

#Read camera image (should automaticcaly save one as well)

inputTxt = ''
while inputTxt != 'q':
    _, img = cam.read()
    print("saved img")
    inputTxt = input("q to quit. Anything to take picture: ").strip().lower()