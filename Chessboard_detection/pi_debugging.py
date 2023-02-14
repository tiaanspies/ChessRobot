import os
import cv2 as cv

def saveImg(img, name):
    dirPath = os.path.dirname(os.path.realpath(__file__))
    relPath = "/TestImages/Temp"
    cv.imwrite(dirPath+relPath+"/"+name, img)