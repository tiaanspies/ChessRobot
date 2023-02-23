import os
import cv2 as cv

def saveImg(img, name):
    dirPath = os.path.dirname(os.path.realpath(__file__))
    relPath = "/TestImages/Temp"
    img1 = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    cv.imwrite(dirPath+relPath+"/"+name, img)