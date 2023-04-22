import os
import cv2 as cv
import numpy as np
from pathlib import Path
from datetime import date
import platform

def saveTempImg(img, name):
    dirPath = os.path.dirname(os.path.realpath(__file__))
    relPath = Path("TestImages","Temp")
    absPath = Path(dirPath, relPath,name)
    
    img1 = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    cv.imwrite(str(absPath.resolve()), img1)

def showImg(images, variables):

    # don't show on linux
    if platform.system() == "Linux":
        return 0

    # while cv.waitKey(1) != ord('q'):
    img_stacked = np.zeros(shape=(images[0].shape[0], 0))
    for i, img in enumerate(images):
        var_name = [name for name in variables if variables[name] is img][0]
        co_ord = (0, 40)
        mean_val = np.mean(img[0:80, 40:80])

        if mean_val > 125:
            color_in = (0, 0, 0)
            color_out = (255, 255, 255)
        else:
            color_in = (255, 255, 255)
            color_out = (0,0,0)

        img1 = cv.putText(img.copy(), var_name, co_ord, cv.FONT_HERSHEY_SIMPLEX, 1.5, color_out, 16)
        img1 = cv.putText(img1, var_name, co_ord, cv.FONT_HERSHEY_SIMPLEX, 1.5, color_in, 4)

        # img_stacked = cv.hconcat([img_stacked, img1])
        cv.namedWindow(var_name, cv.WINDOW_NORMAL)
        cv.imshow(var_name, img1)
        
    # cv.namedWindow('0', cv.WINDOW_NORMAL)
    # cv.imshow('0', img_stacked)
    cv.waitKey(1)
    # cv.destroyAllWindows()

class imgDBManager:
    """
    Class to manage saving images to correct paths while testing.
    Each time a test is run the images will be saved to consequtive folders
    with the parent as todays date under "TestImages".
    """
    def __init__(self) -> None:
        """
        Finds the initial folder and zeros the img name ID.
        """
        self.imgID = 0
        self.folderPath = self.findStartPath()

    def findStartPath(self):
        """
        Find the path to the last subfolder in todays date folder.
        """
        dateToday = date.today()
        dateTodayStr = dateToday.strftime("%d_%m_%Y")

        pathDate = Path("Chessboard_detection", "TestImages", dateTodayStr)
        
        if not pathDate.exists():
            pathDate.mkdir()
        
        # find the highest date id.
        maxID = 0
        for dir in pathDate.iterdir():
            if int(dir.name) > maxID:
                maxID = int(dir.name)

        maxID += 1

        bottomPath = Path(pathDate, str(maxID))
        bottomPath.mkdir(exist_ok=False)
        
        return bottomPath


    def saveDBImage(self, img):
        """
        Check for folders in the TestImages folder.
        Format Data>Number.
        Add image to the 
        """
        imgPath = Path(self.folderPath, str(self.imgID)+".jpg")

        if imgPath.exists():
            print("Error: Logfile image already exists in: "+str(imgPath.resolve()))
            return

        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        cv.imwrite(str(imgPath.resolve()), img)
        self.imgID += 1


