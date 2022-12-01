import os
import math

import numpy as np
from scipy import signal
from sklearn.cluster import KMeans
import cv2 as cv
from collections import Counter

from matplotlib import pyplot as plt

# from Chessboard_detection import Fake_Camera
import Fake_Camera

CAMERA_RESOLUTION = (640, 480)

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

class ChessBoard:
    def __init__(self, img) -> None:
        """
        Initializes board object and finds position of empty board
        """
        # CAMERA OBJECT PROPERTIES
        self.initialImage = np.zeros(CAMERA_RESOLUTION)

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

        # save the blank boardS
        # self.setInitialImage(camera)
        self.initialImage = img

        # see which blak and white threshold makes the board the easiest to find
        # Opt is estimated by middle of min and max
        thresholdOpt = self.findOptimalThreshold(self.initialImage, onlyFindOne=False, erodeSize=3, blurSize=3)

        _, cornersInt  = self.findBoardCorners(self.initialImage, thresholdOpt)

        cornersIntReshaped = np.reshape(cornersInt, self.BOARD_SIZE_INT + (2,))
        cornersIntReoriented = self.makeTopRowFirst(cornersIntReshaped)

        self.cornersExt = self.estimateExternalCorners(cornersIntReoriented)
        self.flipBoard = False       
        
        # # # ====== Uncomment this code to draw chessboardCorners
        # cornersNew = np.reshape(self.cornersExt, (81, 2))
        # img_new = cv.drawChessboardCorners(img, self.BOARD_SIZE, cornersNew, True)
        # showImg(img_new)

    def findClusterImg(self, img):
        """
        Assigns pixels to their closest cluster.
        returns image with all pixels assigned to cluster
        """
        # maskedImage = self.maskImage(img)

        imgReshaped = np.reshape(img, (img.shape[0]*img.shape[1], 3))
        transformed = self.HSVTransform(imgReshaped)

        predictions = self.kmeans.predict(transformed)
        clustersInt = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]])

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
            transformed = self.HSVTransform(imgReshaped)
            predictionReshaped = self.kmeans.predict(transformed)
            predictions[id] = np.reshape(predictionReshaped, (block.shape[0], block.shape[1]))

        return predictions
    
    def initBoardWithStartPos(self, img):
        """ 
        Fit the K-Means cluster with all pieces on starting squares.
        ALso finds which color is at the top.
        
        """
        self.fitKClusters(img, weighted=False)

        positions = self.getCurrentPositions(img)

        meanBottom = np.round(np.mean(positions[-2:, :], axis=(0, 1)))
        meanTop = np.round(np.mean(positions[:2, :]))

        if meanTop == -1:
            self.flipBoard = True

        # Check if starting position is valid with all pieces on starting squares
        topCorrect = (positions.flatten()[:16] == meanTop).all()
        middleCorrect = (positions.flatten()[16:48] == 0).all()
        bottomCorrect = (positions.flatten()[48:] == -1*meanTop).all()
        if not topCorrect or not middleCorrect or not bottomCorrect:
            raise("starting position is incorrect")

        humanColor = meanTop
        robotColor = meanBottom

        return humanColor, robotColor

    def HSVTransform(self, HSVImg):
        

        maxSat = np.max(HSVImg[:, 1])
        maxI = np.max(HSVImg[:, 2])

        satThreshold = maxSat - maxSat*0.8*HSVImg[:, 2]/maxI

        threshPoints = HSVImg[:, 1] > satThreshold
        count = np.count_nonzero(threshPoints)
        transformedCol = np.zeros(shape=(HSVImg.shape[0],1))
        transformedCol[threshPoints, 0] = HSVImg[threshPoints, 0]

        threshPoints = np.invert(threshPoints)
        newHSV = np.reshape(HSVImg, (256, 256, 3))
        newImg = cv.cvtColor(newHSV, cv.COLOR_HSV2RGB)

        showImg(newImg)

        newImg = np.reshape(newImg, (256*256, 3))
        newImg[threshPoints] = np.array([255,255,255])
        newImg = np.reshape(newImg, (256, 256, 3))
        showImg(newImg)
        
        transformedCol[threshPoints, 0] = HSVImg[threshPoints, 2]/10+200

        return transformedCol

    def fitKClusters(self, img, weighted=False):
        """
        Fit 4 k-means clusters to the image. Use HSV color scale
        Weighting can be used if pieces are on starting squares.
        To increase the weight of pieces.
        """

        # Mask image by turning all pixels outside of board black
        maskedImage = self.maskImage(img)
        maskedImageHSV = cv.cvtColor(maskedImage, cv.COLOR_RGB2HSV)

        # Get a block for each square and resize it to 32x32,
        # then stack back together for 256x256 image
        blocks = self.splitBoardIntoSquares(maskedImageHSV)
        resizedImg = np.reshape(blocks, (8, 8, 32, 32, 3))
        resizedImg = np.hstack(resizedImg)
        resizedImg = np.hstack(resizedImg)

        blur = cv.medianBlur(resizedImg, 7)
        
        # reshape image into a single line for k means fitting
        self.kmeans = KMeans(n_clusters=4)
        imgReshaped = np.reshape(blur, (blur.shape[0]*blur.shape[1], 3))
        transformed = self.HSVTransform(imgReshaped)

        # if weighed is true apply a gaussian weight to each block.
        # add priority to blocks with peices on (starting squares)
        if weighted:
            kern = gkern(int(np.shape(blocks[0])[0]), self.KERN_STD)
            kernArr = np.tile(kern, (8, 8))
            kernArr[:2*32, :8*32] *= 2
            kernArr[-2*32:, -8*32:] *= 2
        
            kernReshaped = np.reshape(kernArr, (blur.shape[0]*blur.shape[1]))
            self.kmeans.fit(imgReshaped, sample_weight=kernReshaped)
        else:
            self.kmeans.fit(transformed)       
        

        img2 = self.findClusterImg(self.maskImage(img))
        plt.imshow(img2)
        # showImg(img2)

        #Assign cluster id to all pixels on blocks
        # find what cluster id is black or white
        blockClusterID = self.findBlockClusters(blocks)
        self.findPieceID(blockClusterID)

        return None
    
    def setInitialImage(self, camera, confirmFirst=False):
        """
        If confirmFirst then it will ask the user for confirmation 
        on whether they want to initialize on that image.
        Otherwise just use image immediatly
        """
        while True:
            #Get image frame
            ret, frame = camera.read()

            if not ret:
                print("Can't receive frame (stream end?).")
                raise("Could not receive frame from camera to set initial image")
            
            # Display video
            if confirmFirst:
                cv.imshow('Initial Image', frame)

                # Pause video when q is pressed
                if cv.waitKey(1) == ord('q'):
                    resp = input("Are you happy with this initialization image? (y/n)")
                
                    # Check if the image is good enough
                    if resp == 'y':
                        cv.destroyWindow('Initial Image')
                        break

            if not confirmFirst:
                break
        one = np.ones((8, 8), dtype=np.int32)
        self.initialImage = frame.copy()
        

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

    def threshHoldAndFindBoard(self, grayBlurImg, threshold, erodeSize):
        """
        Theshold image and invert as detect chessboard requires a white border around chessboard.
        """
        # Threshold
        _, thresh = cv.threshold(grayBlurImg, threshold, 255, cv.THRESH_BINARY_INV)
        
        # If erodeSize is invalid skip eroding and dilating
        if erodeSize > 0:
            element = cv.getStructuringElement(cv.MORPH_RECT, (erodeSize, erodeSize))
            erode = cv.erode(thresh, element)
            dilate = cv.dilate(erode, element)
        else:
            dilate = thresh

        # find chessboard corners, If its too slow add the fast stop param
        didFind, intCorners = cv.findChessboardCorners(dilate, self.BOARD_SIZE_INT)

        return didFind, intCorners

    def findBoardCorners(self, img, threshold, blursize=3, erodeSize=3):
        grayBlur = self.findGrayBlur(img, blursize)
        ret, corners = self.threshHoldAndFindBoard(grayBlur, threshold, erodeSize)
        return ret, corners

    def findOptimalThreshold(self, img, blurSize=3, erodeSize=3, onlyFindOne=False):
        stepSize = 10

        #Find gray blurry img
        grayBlur = self.findGrayBlur(img, blurSize)

        #do a first rough pass by checking all values with step size
        # There is no point in checking 0 and 255
        result_array = [0]*(math.ceil(255/stepSize)-2)
        for i in range(1, math.floor(255/stepSize)):
            result_array[i-1], _ = self.threshHoldAndFindBoard(grayBlur, i*stepSize, erodeSize)

            # stop at first success if flag to find one is true
            if onlyFindOne and result_array[i-1] == True:
                return i*stepSize

        #Find min and max threshold bound
        respMin, minBound = minPos(result_array)
        minBound = stepSize + minBound*stepSize - 1
        respMax, maxBound = maxPos(result_array)
        maxBound = stepSize + maxBound*stepSize + 1

        if not respMin or not respMax:
            print("Could not find successful threshold")
            exit()

        # finetune min
        iter = 0
        result = False
        while iter < 10:
            result, _ = self.threshHoldAndFindBoard(grayBlur, minBound, erodeSize)
            if not result:
                break
            minBound -= 1
        else:
            raise("minbound never found finetuned val")

        # finetune max
        result = False
        while iter < 10:
            result, _ = self.threshHoldAndFindBoard(grayBlur, maxBound, erodeSize)
            if not result:
                break
            maxBound += 1
        else:
            raise("maxBound never found finetuned val")

        # return average of minBound and Maxbound to hopefully be most reliable threshold
        thresholdOpt = int((maxBound+minBound)/2)
        return thresholdOpt 

    def estimateExternalCorners(self, cornersInt):
        """
        Expand the internal corners by one row. 
        """

        diffRow = np.diff(cornersInt, axis=0)
        meanDiffRow = np.mean(diffRow, axis=0)
        externTop = 2*cornersInt[None, 0, :, :] - cornersInt[None, 1, :, :]
        externBottom = 2*cornersInt[None, -1, :, :] - cornersInt[None, -2, :, :]

        difCol = np.diff(cornersInt, axis=1)
        meanDiffCol = np.mean(difCol, axis=1)
        meanDiffCol = np.reshape(meanDiffCol, (7, 1, 2))

        externleft = 2*cornersInt[:, None, 0, :] - cornersInt[:, None, 1, :]
        externRight = 2*cornersInt[:, None, -1, :] - cornersInt[:, None, -2, :]

        topLeft = 3*cornersInt[None, 0, None, 0, :] - cornersInt[None, 1, None, 0, :] - cornersInt[None, 0, None, 1, :]
        topRight = 3*cornersInt[None, 0, None, -1, :] - cornersInt[None, 1, None, -1, :] - cornersInt[None, 0, None, -2, :]
        botLeft = 3*cornersInt[None, -1, None, 0, :] - cornersInt[None, -2, None, 0, :] - cornersInt[None, -1, None, 1, :]
        botRight = 3*cornersInt[None, -1, None, -1, :] - cornersInt[None, -2, None, -1, :] - cornersInt[None, -1, None, -2, :]

        cornersExtRow0 = np.hstack([topLeft, externTop, topRight])
        cornersExtRow1 = np.hstack([externleft, cornersInt, externRight])
        cornersExtRow2 = np.hstack([botLeft, externBottom, botRight])

        cornersExt = np.vstack([cornersExtRow0, cornersExtRow1, cornersExtRow2])

        return cornersExt
    
    def findCentroidsOfCorners(self, corners):
        row_top_pt_1 = corners[0][0]
        row_top_pt_2 = corners[0][-1]
        centroid_top = (row_top_pt_1 + row_top_pt_2) / 2

        row_left_pt_1 = corners[0][0]
        row_left_pt_2 = corners[-1][0]
        centroid_left = (row_left_pt_1 + row_left_pt_2) / 2

        row_right_pt_1 = corners[0][-1]
        row_right_pt_2 = corners[-1][-1]
        centroid_right = (row_right_pt_1 + row_right_pt_2) / 2

        row_bottom_pt_1 = corners[-1][0]
        row_bottom_pt_2 = corners[-1][-1]
        centroid_bottom = (row_bottom_pt_1 + row_bottom_pt_2) / 2

        return centroid_top, centroid_left, centroid_right, centroid_bottom

    def makeTopRowFirst(self, corners):
        # TODO: adjust angle to keep top row at the top of image
        # The array should start form top left.
        # top row is the top row of array. We are aiming to make it the 
        # top row of the image as well
        
        top, left, right, bottom = self.findCentroidsOfCorners(corners)

        if top[1] > np.min([left[1], right[1], bottom[1]]) \
            and top[1] < np.max([left[1], right[1], bottom[1]]):
            # top row is between. therefore it must be on the side of the image.
            # transpose, and flip it.
            newArr = np.transpose(corners, axes=(1, 0, 2))
        else:
            newArr = corners
        
        top, left, right, bottom = self.findCentroidsOfCorners(newArr)
        
        if top[1] > np.max([left[1], right[1], bottom[1]]):
            #Row is at bottom of all. flip to make top.
            newArr = np.flip(newArr, axis=0)
        
        diff = newArr[0][-1] - newArr[0][0]
        firstRowAngle = np.arctan2(diff[1], diff[0])
        firstRowAngle = (firstRowAngle + 2*np.pi) % (2*np.pi)
        if firstRowAngle > np.pi/2 and firstRowAngle < (4/3*np.pi):
            newArr = np.flip(newArr, axis=1)

        return newArr
    
    def maskImage(self, img):
        """
        Create a mask so that only the chessboard is visible.
        Uses the 4 furthest corners for mask edge.
        """
        outerCorners = np.vstack([
                self.cornersExt[None, 0, 0, :],
                self.cornersExt[None, 0, -1, :],
                self.cornersExt[None, -1, -1, :],
                self.cornersExt[None, -1, 0, :]
            ])

        mask = np.zeros(img.shape, dtype=np.uint8)
        mask = cv.fillConvexPoly(mask, np.int32(outerCorners), color=(255, 255, 255))

        maskedImage = cv.bitwise_and(img, mask)
        return maskedImage

    def getCurrentPositions(self, img):
        """
        Look at each square in board and see which color piece is on it.
        """
        # Read new image from camera object
        hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
        blurredHSV = cv.medianBlur(hsv, 11)
        
        
        ### DEBUG TO SHOW CLUSTER IMAGE
        masked = self.maskImage(blurredHSV)
        cluster = self.findClusterImg(masked)
        img_new = cv.cvtColor(cluster, cv.COLOR_HSV2RGB)
        showImg(img_new)
        ## END DEBUG
        

        blocks = self.splitBoardIntoSquares(blurredHSV)
        clustered = self.findBlockClusters(blocks)
        blockIDs = self.detectPieces(clustered)

        blockIDs = np.reshape(blockIDs, (8, 8))

        if self.flipBoard:
            blockIDs = np.flip(blockIDs, axis=0)
            blockIDs = np.flip(blockIDs, axis=1)
            
        return blockIDs

    def splitBoardIntoSquares(self, img):
        """
        Take a img of the chessboard with vertex positions and split
        each square into a seperate array.
        """
        blocks = []
        for r in range(self.BOARD_SIZE_INT[0] + 1):
            for c in range(self.BOARD_SIZE_INT[1] + 1):
                # minX = np.min([self.cornersExt[r, c, 0], self.cornersExt[r+1, c, 0]])
                # minY = np.min([self.cornersExt[r, c, 0], self.cornersExt[r, c+1, 0]])

                # maxX = np.max([self.cornersExt[r, c+1, 0], self.cornersExt[r+1, c+1, 0]])
                # maxY = np.max([self.cornersExt[r+1, c, 0], self.cornersExt[r+1, c+1, 0]])

                vertices = self.cornersExt[r:r+2, c:c+2, :].astype(np.uint)
                minX = np.min(vertices[:, :, 0], axis=(0,1))
                minY = np.min(vertices[:, :, 1], axis=(0,1))
                
                maxX = np.max(vertices[:, :, 0], axis=(0,1))
                maxY = np.max(vertices[:, :, 1], axis=(0,1))

                block = img[minY:maxY, minX:maxX]
                block = cv.resize(block, (32, 32))

                # img = cv.rectangle(img, [minX, minY], [maxX, maxY], (255, 0, 0))

                blocks.append(block)

        return blocks
    
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

        blocksize = blocks.shape[0] # assuming pieces are square
        kern = gkern(kernlen=int(blocksize/2), std=self.KERN_STD)
        
        pieces = np.zeros((np.shape(blocks)[0]), dtype=np.int8)

        for id, block in enumerate(blocks):
            pieceID = self.detectPiece(block, kern)
            if pieceID == self.whiteID:
                pieces[id] = 1
            elif pieceID == self.blackID:
                pieces[id] = -1
            else:
                pieces[id] = 0

        return pieces

    def findPieceID(self, blocks):
        """
        Assign find which cluster the black and white pieces
        belong to.
        """
        topPieces = np.zeros((self.BOARD_SIZE_INT[0]+1)*2)
        bottomPieces = np.zeros((self.BOARD_SIZE_INT[0]+1)*2)

        kern = gkern(kernlen=int(np.shape(blocks)[0]/2), std=self.KERN_STD)

        for id, block in enumerate(blocks[0:16]):
            topPieces[id] = self.detectPiece(block, kern)

        for id, block in enumerate(blocks[-17:-1]):
            bottomPieces[id] = self.detectPiece(block, kern)

        topPieceMax = (Counter(topPieces).most_common(1))[0][0]
        bottomPieceMax = (Counter(bottomPieces).most_common(1))[0][0]

        img = np.zeros(shape=(1,2,3))
        img[0, 0] = np.array(self.kmeans.cluster_centers_[int(topPieceMax)])
        img[0, 1] = np.array(self.kmeans.cluster_centers_[int(bottomPieceMax)])

        imgGray = cv.cvtColor(img.astype(np.uint8), cv.COLOR_BGR2GRAY)
        if imgGray[0, 0] > imgGray[0, 1]:
            self.whiteID = int(topPieceMax)
            self.blackID = int(bottomPieceMax)
        else:
            self.whiteID = int(bottomPieceMax)
            self.blackID = int(topPieceMax)
 
def showImg(*images):
    while cv.waitKey(1) != ord('q'):
        for i, img in enumerate(images):
            # cv.namedWindow(str(i), cv.WINDOW_NORMAL)
            
            cv.imshow(str(i), img)


def main():
    # Open Video camera
    # cam = cv.VideoCapture(0)
    dirPath = os.path.dirname(os.path.realpath(__file__))
    relPath = "\\TestImages\\Test_Set_1"
    cam = Fake_Camera.FakeCamera(CAMERA_RESOLUTION, dirPath + relPath)    

    if not cam.isOpened():
        raise("Cannot open camera.")

    # Initialize ChessBoard object and select optimal thresh
    # Board must be empty when this is called
    s, img = cam.read()
    board = ChessBoard(img)

    # NB --- Board is setup in starting setup.
    # Runs kmeans clustering to group peice and board colours
    s, img = cam.read()
    board.fitKClusters(img, False)

    # display video of chessboard with corners
    s, img = cam.read()  
    while s is True:  
        positions = board.getCurrentPositions(img)
        s, img = cam.read()  
        print(positions)

    cam.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()