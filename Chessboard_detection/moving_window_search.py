import os
import math
import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans
from scipy import signal
from collections import Counter
import Camera.Camera_Manager as Camera_Manager
from matplotlib import pyplot as plt

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
    
    def initBoardWithStartPos(self, img):
        self.fitKClusters(img, weighted=True)

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
        self.kmeans = KMeans(n_clusters=4, init='k-means++', n_init=1)
        imgReshaped = np.reshape(blur, (blur.shape[0]*blur.shape[1], 3))

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
            self.kmeans.fit(imgReshaped)       
        
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
        print("Finding Threshold")
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
        print("Opt Thresh:", thresholdOpt)
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

        debug.saveImg(img_new, "test.jpg")
        ## END DEBUG

        blocks = self.splitBoardIntoSquares(blurredHSV)
        clustered = self.findBlockClusters(blocks)
        blockIDs = self.detectPieces(clustered)

        blockIDs = np.reshape(blockIDs, (8, 8))

        if self.flipBoard:
            blockIDs = np.flip(blockIDs, axis=0)
            blockIDs = np.flip(blockIDs, axis=1)
        
        print(blockIDs)
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

def findClusterImg(kmeans, img):
    """
    Assigns pixels to their closest cluster.
    returns image with all pixels assigned to cluster
    """
    imgReshaped = np.reshape(img, (img.shape[0]*img.shape[1], 3))

    predictions = kmeans.predict(imgReshaped)
    clustersInt = kmeans.cluster_centers_.astype(np.uint8)

    newImg = [clustersInt[x] for x in predictions]
    newImg = np.reshape(newImg, (img.shape[0], img.shape[1], 3))
    
    # return cv.cvtColor(newImg, cv.COLOR_HSV2RGB)
    # return cv.cvtColor(newImg, cv.COLOR_Lab2RGB)
    return newImg

def cannyShapes(img):
    cannyEdges = cv.Canny(img,100,200)
    cannyEdges = cv.dilate(cannyEdges, (5,5))

    # ret, thresh = cv.threshold(img, 127, 255, 0)
    # contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # i=0
    # newContours = []
    # max = 0
    # for contour in contours:
    
    #     # here we are ignoring first counter because 
    #     # findcontour function detects whole image as shape
    #     if i == 0:
    #         i = 1
    #         continue
    
    #     # cv2.approxPloyDP() function to approximate the shape
    #     approx = cv.approxPolyDP(
    #         contour, 0.05 * cv.arcLength(contour, True), True)
        
    #     area = cv.contourArea(contour)
    #     if area > max:
    #         max = area
    #     # putting shape name at center of each shape

    #     if len(approx) == 4 and area > 2000:
    #         newContours.append(contour)

        
    # print("Max area: ", max)

    # img = cv.drawContours(img, newContours, -1, (0,255,0), 2)
    # cv.imshow("1", thresh)

    cv.imshow("Canny edges dialted", cannyEdges)
    cv.waitKey()
    cv.destroyAllWindows()

def findContEdges(img):
    ret, thresh = cv.threshold(img, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    color = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    i=0
    newContours = []
    max = 0
    for contour in contours:
    
        # here we are ignoring first counter because 
        # findcontour function detects whole image as shape
        if i == 0:
            i = 1
            continue
    
        # cv2.approxPloyDP() function to approximate the shape
        approx = cv.approxPolyDP(
            contour, 0.05 * cv.arcLength(contour, True), True)
        
        area = cv.contourArea(contour)
        if area > max:
            max = area
        # putting shape name at center of each shape

        if len(approx) == 4 and area > 2000:
            newContours.append(contour)

        
    print("Max area: ", max)

    img = cv.drawContours(color, newContours, -1, (0,255,0), 2)
    cv.imshow("1", thresh)
    cv.waitKey()
    cv.destroyAllWindows()

def otsu(img):
     # find normalized_histogram, and its cumulative distribution function
    hist = cv.calcHist([img],[0],None,[256],[0,256])
    hist_norm = hist.ravel()/hist.sum()
    Q = hist_norm.cumsum()
    bins = np.arange(256)
    fn_min = np.inf
    thresh = -1
    for i in range(1,256):
        p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
        q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
        if q1 < 1.e-6 or q2 < 1.e-6:
            continue
        b1,b2 = np.hsplit(bins,[i]) # weights
        # finding means and variances
        m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
        v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
        # calculates the minimization function
        fn = v1*q1 + v2*q2
        if fn < fn_min:
            fn_min = fn
            thresh = i
    # find otsu's threshold value with OpenCV function
    ret, otsu = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    print( "{} {}".format(thresh,ret) )
    return otsu

def showImg(images):
    while cv.waitKey(1) != ord('q'):
        for i, img in enumerate(images):
            cv.namedWindow(str(i), cv.WINDOW_NORMAL)
            
            cv.imshow(str(i), img)

def main():
    # find path for where test images will be read
    dirPath = os.path.dirname(os.path.realpath(__file__))
    relPath = "\\TestImages\\Set_2_W_Only"

    # read test image
    cam = Camera_Manager.FakeCamera(CAMERA_RESOLUTION, dirPath + relPath) 
    s, img = cam.read()   

    if not cam.isOpened():
        raise("Cannot open camera.")

    # img = cv.imread('Chessboard_detection\TestImages\Temp\\empty.JPG')
    board = ChessBoard(img)

    # get the edges of chess board that got calculated from empty chess board
    corner0 = board.cornersExt[0, 0, :].astype(np.int32)
    corner1 = board.cornersExt[-1, 0, :].astype(np.int32)
    corner2 = board.cornersExt[-1, -1, :].astype(np.int32)
    corner3 = board.cornersExt[0, -1, :].astype(np.int32)

    # draw circles on the corners of the chessboard
    imgCircle = cv.circle(img, corner0, radius=2, color=(0, 0, 255), thickness=-1)
    imgCircle = cv.circle(img, corner1, radius=2, color=(0, 0, 255), thickness=-1)
    imgCircle = cv.circle(img, corner2, radius=2, color=(0, 0, 255), thickness=-1)
    imgCircle = cv.circle(img, corner3, radius=2, color=(0, 0, 255), thickness=-1)

    # draw a bounding box around the outer corners
    # this basically puts an aligned square around everything
    # would be better to use the exact points and stretch it into a square
    x,y,w,h = cv.boundingRect(np.array([corner0, corner1, corner2, corner3]))

    # different methods for comparing the accuracy of template matching
    # from what I saw the TM_CCOEFF works the best. It is the only one
    # that is able to match an empty chessboard with one with pieces on.
    # script runs through all the different methods in methods array

    # methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
    #         'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
    methods = ['cv.TM_CCOEFF']

    while s:
        # take cropped image of chessboard from previous image where it was found.
        imgCropped = img[y:y+h, x:x+w]
        cv.imshow('cropped', imgCropped)

        # read new image and roll it to move the chessboard around randomly
        s, img = cam.read()
        x_rand = np.random.randint(-40, 40)
        y_rand = np.random.randint(-100, 100)
        print("Roll Amounts:", x_rand, y_rand)
        img = np.roll(img, [y_rand, x_rand], axis=(0,1))

        # find image using different methods
        for meth in methods:
            img1 = img.copy()
            method = eval(meth)
            # Apply template Matching
            res = cv.matchTemplate(img1,imgCropped,method)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
            if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc
            x, y = top_left
            bottom_right = (top_left[0] + w, top_left[1] + h)

            # plot results
            cv.rectangle(img1,top_left, bottom_right, 255, 2)
            plt.subplot(121),plt.imshow(res,cmap = 'gray')
            plt.title('Matching Result'), plt.xticks([]), plt.yticks([])

            # Draw a circle in the top left of matching area
            img1 = cv.circle(img1, top_left, radius=4, color=(255, 255, 255), thickness=-1)
            plt.subplot(122),plt.imshow(img1,cmap = 'gray')
            plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
            plt.suptitle(meth)
            plt.show()



if __name__ == "__main__":
    main()