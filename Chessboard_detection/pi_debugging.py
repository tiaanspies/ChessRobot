import os
import cv2 as cv
from pathlib import Path
from datetime import date

def saveTempImg(img, name):
    dirPath = os.path.dirname(os.path.realpath(__file__))
    relPath = "/TestImages/Temp"
    img1 = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    cv.imwrite(dirPath+relPath+"/"+name, img1)

def showImg(images):
    while cv.waitKey(1) != ord('q'):
        for i, img in enumerate(images):
            cv.namedWindow(str(i), cv.WINDOW_NORMAL)
            
            cv.imshow(str(i), img)

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


