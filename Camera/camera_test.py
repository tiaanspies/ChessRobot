import os
from Camera import Camera_Manager
from Chessboard_detection import pi_debugging
from time import sleep

# start camera
dirPath = os.path.dirname(os.path.realpath(__file__))
relPath = "/Chessboard_detection/TestImages/Temp"
cam = Camera_Manager.RPiCamera(
        res=(480, 640), 
        absPath=dirPath+relPath, 
        storeImgHist=True, 
        loadSavedFirst=False
    )

# Change storeImgHist to automatically save images while running

inputTxt = ''
while inputTxt != 'q':
    _, img = cam.read()
    print("saved img")
    # inputTxt = input("q to quit. Anything to take picture: ").strip().lower()
    sleep(1)