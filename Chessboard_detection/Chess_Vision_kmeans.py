import os
import math

import numpy as np
from scipy import signal
from sklearn.cluster import KMeans
import cv2 as cv
from collections import Counter

try:
    from Chessboard_detection import pi_debugging as debug
    from Camera import Camera_Manager
except ModuleNotFoundError:
    import pi_debugging as debug
    import Camera.Camera_Manager as Camera_Manager
# import Fake_Camera

# CAMERA_RESOLUTION = (640, 480)
CAMERA_RESOLUTION = (480, 640)

def empty(a):
    pass

def minPos(arr):
    """
    Finds the first occurance of a True array element
    in a boolean array.
    """
    found = False
    index = 0
    for i, item in enumerate(arr):
        if item:
            index = i
            found = True
            break

    return found, index

def maxPos(arr):
    """
    Finds the last occurance of a True array element
    in a boolean array.
    """
    found = False
    index = 0
    for i, item in reversed(list(enumerate(arr))):
        if item:
            index = abs(i)
            found = True
            break

    return found, index

def gkern(kernlen=21, std=3):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

class ChessBoardClassifier:
    def __init__(self, blurred_hsv_squares) -> None:
        """
        Initializes board object and finds position of empty board
        """
        # BOARD SIZE PROPERTIES
        # KERN STD is the standard deviation of the gaussian kernal used for 
        # weighting the piece prediction. 
        self.BOARD_SIZE = (9, 9)
        self.BOARD_SIZE_INT = (7, 7)
        self.KERN_STD = 6
        self.PIECE_WEIGHT = 3

        # Kmeans class with the id of the black and white pieces
        self.kmeans = None
        self.whiteID = None
        self.blackID = None

        self.initBoardWithStartPos(blurred_hsv_squares)

    def findClusterImg(self, img):
        """
        Assigns pixels to their closest cluster.
        returns image with all pixels assigned to cluster
        """
        # maskedImage = self.maskImage(img)

        imgReshaped = np.reshape(img, (img.shape[0]*img.shape[1], 3))

        predictions = self.kmeans.predict(imgReshaped)
        clustersInt = self.kmeans.cluster_centers_.astype(np.uint8)

        newImg = [clustersInt[x] for x in predictions]
        newImg = np.reshape(newImg, (img.shape[0], img.shape[1], 3))
        
        return newImg

    def findBlockClusters(self, blocks):
        """
        Receives blocks of 32x32 of pixel color values.
        assigns each pixel to a cluster and returns cluster ID
        return shape is (NrBlocks, BlockRows, BlockCols)
        """
        blockShape = np.shape(blocks)
        predictions = np.zeros(shape=(blockShape[0], blockShape[1], blockShape[2]), dtype=np.uint8)

        for id, block in enumerate(blocks):
            imgReshaped = np.reshape(block, (block.shape[0]*block.shape[1], 3))
            predictionReshaped = self.kmeans.predict(imgReshaped)
            predictions[id] = np.reshape(predictionReshaped, (block.shape[0], block.shape[1]))

        return predictions
    
    def initBoardWithStartPos(self, blurred_hsv_squares):
        self.fitKClusters(blurred_hsv_squares, weighted=True)

        positions = self.getCurrentPositions(blurred_hsv_squares)

        return positions


    def fitKClusters(self, blurred_hsv_squares:np.ndarray, weighted=False):
        """
        Fit 4 k-means clusters to the image. Use HSV color scale
        Weighting can be used if pieces are on starting squares.
        To increase the weight of pieces.
        """
        # reshape image into a single line for k means fitting
        self.kmeans = KMeans(n_clusters=4, init='k-means++', n_init=1)
        n, h, w, c = blurred_hsv_squares.shape
        imgReshaped = np.reshape(blurred_hsv_squares, (n * h * w, c))

        # if weighed is true apply a gaussian weight to each block.
        # add priority to blocks with peices on (starting squares)
        if weighted:
            kern = gkern(h, self.KERN_STD)
            kernArr = np.tile(kern, (n))
        
            kernReshaped = kernArr.flatten()
            self.kmeans.fit(imgReshaped, sample_weight=kernReshaped)
        else:
            self.kmeans.fit(imgReshaped)       
        
        img = self.findClusterImg(blurred_hsv_squares.reshape(n*h, w, c))
        debug.saveTempImg(cv.cvtColor(img, cv.COLOR_HSV2BGR), "cluster.jpg")
        #Assign cluster id to all pixels on blocks
        # find what cluster id is black or white
        blockClusterID = self.findBlockClusters(blurred_hsv_squares)
        self.setBlackWhiteIDs(blockClusterID)


    def findGrayBlur(self, img, blurSize):
        # Convert img to grayscale
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # If blursize is invalid then skip blurring
        if blurSize > 0:
            # blur = cv.blur(gray, (blurSize, blurSize))
            blur = cv.medianBlur(gray, blurSize)
        else:
            blur = gray

        return blur

    def getCurrentPositions(self, blocks):
        """
        Look at each square in board and see which color piece is on it.
        """
        
        ### DEBUG TO SHOW CLUSTER IMAGE
        cluster = self.findClusterImg(blocks[0])
        img_new = cv.cvtColor(cluster, cv.COLOR_HSV2BGR)

        debug.saveTempImg(img_new, "test.jpg")
        ## END DEBUG
        
        # raise NotImplementedError
        clustered = self.findBlockClusters(blocks)
        blockIDs = self.detectPieces(clustered)
        
        print(blockIDs)
        return blockIDs
    
    def detectPiece(self, block, kern):
        """
        Given a block it decides which cluster it belongs to.
        Higher weights are placed on pixels closer to the 
        center of the block, as well as to pieces.
        """
        
        scores = np.zeros((4))

        for r, row in enumerate(block):
            for c, pixel in enumerate(row):
                scores [pixel] += kern[r, c]

        if self.whiteID is not None and self.blackID is not None:
            scores[self.whiteID] *= self.PIECE_WEIGHT
            scores[self.blackID] *= self.PIECE_WEIGHT

        return np.argmax(scores, axis = 0)


    def detectPieces(self, blocks):
        """
        given a square, detect which cluster is likely present in the center.
        """

        n, h, w = blocks.shape # assuming pieces are square
        kern = gkern(kernlen=h, std=self.KERN_STD)
        
        pieces = np.zeros((n), dtype=np.int8)

        for id, block in enumerate(blocks):
            pieceID = self.detectPiece(block, kern)
            if pieceID == self.whiteID:
                pieces[id] = 1
            elif pieceID == self.blackID:
                pieces[id] = -1
            else:
                pieces[id] = 0

        return pieces

    def setBlackWhiteIDs(self, blocks):
        """
        Assign find which cluster the black and white pieces
        belong to.
        """
        topPieces = np.zeros((self.BOARD_SIZE_INT[0]+1)*2)
        bottomPieces = np.zeros((self.BOARD_SIZE_INT[0]+1)*2)

        n, h, w = blocks.shape

        kern = gkern(kernlen=h, std=self.KERN_STD)

        for id, block in enumerate(blocks[0:16]):
            topPieces[id] = self.detectPiece(block, kern)

        for id, block in enumerate(blocks[-17:-1]):
            bottomPieces[id] = self.detectPiece(block, kern)

        topPieceMax = (Counter(topPieces).most_common(1))[0][0]
        bottomPieceMax = (Counter(bottomPieces).most_common(1))[0][0]

        # img = np.zeros(shape=(1,2,3))
        # img[0, 0] = np.array(self.kmeans.cluster_centers_[int(topPieceMax)])
        # img[0, 1] = np.array(self.kmeans.cluster_centers_[int(bottomPieceMax)])

        # imgGray = cv.cvtColor(img.astype(np.uint8), cv.COLOR_BGR2GRAY)
        # if imgGray[0, 0] > imgGray[0, 1]:
        #     self.whiteID = int(topPieceMax)
        #     self.blackID = int(bottomPieceMax)
        # else:
        #     self.whiteID = int(bottomPieceMax)
        #     self.blackID = int(topPieceMax)

        # TODO: add decting color functionality. 
        # TODO: for now assuming white is at the bottom and black is at the top

        self.whiteID = int(bottomPieceMax)
        self.blackID = int(topPieceMax)
        
 
def showImg(img):
    # while cv.waitKey(1) != ord('q'):
    # for i, img in enumerate(images):
    cv.namedWindow("1", cv.WINDOW_NORMAL)
    
    cv.imshow("1", img)


def main():
    pass

if __name__ == "__main__":
    main()