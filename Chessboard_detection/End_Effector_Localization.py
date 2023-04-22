import Camera_Manager, platform, threading, queue
from pathlib import Path
import cv2 as cv
import pi_debugging as debug
import numpy as np
import matplotlib.pyplot as plt
import heapq, time

import sys
sys.path.append('./')
from IK_Solvers.traditional import ChessMoves
from motor_commands import MotorCommands


xflip = 0
yflip = 0

class matchEngine:
    """ Scale invariant feature transform manager"""
    def __init__(self, matcherType='SIFT') -> None:
        """ Can use SIFT or ORB matching, sift is more accurate but requires a lot
        more runtime ~5x increase. 
        """
        if matcherType=='SIFT':
            self.matcher = cv.SIFT_create()
            self.bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
        elif matcherType=='ORB':
            self.matcher = cv.ORB_create()
            self.bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        else:
            raise("Invalid matrhcerType, currently support'SIFT' and 'ORB'.")
        
        # initialize descriptors and key points
        self.kpCurrent = None
        self.des = None
    
DEBUG_LEVEL = 0
MATCHES_TO_KEEP = 200

def detectAccurateMatches(img1, img2, descriptor, matcher):
    key_points1, des1 = descriptor.detectAndCompute(img1,None)
    key_Points2, des2 = descriptor.detectAndCompute(img2,None)

    # debug.showImg([img1, img2], locals())
    # cv.waitKey(0)
    debug.saveTempImg(img1, "img1_matches.png")
    debug.saveTempImg(img2, "img2_matches.png")

    if DEBUG_LEVEL > 1:
        img4=cv.drawKeypoints(img1,key_points1,img2,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        debug.showImg([img4], locals())

    # Calculate the best matches for each point key_point pair.
    matches = matcher.match(des1,des2)

    # filter out best matches.
    # filter by distance
    # valid_matches = np.array([x for x in matches if x.distance < 50])

    # filter by portion of best matches
    valid_matches = heapq.nsmallest(MATCHES_TO_KEEP, matches, key = lambda x: x.distance)
    valid_matches = np.array(valid_matches)
    print("Number of matches: ", len(valid_matches))

    matched_image = cv.drawMatches(img1, key_points1, img2, key_Points2, valid_matches, None, flags=2)
    debug.showImg([matched_image], locals())

    # TODO: #2 Write better function for selecting the number of features to keep

    # Get the co-ordinated of the selected matches
    point1_coords = np.zeros((len(valid_matches), 2), dtype=np.float32)
    point2_coords = np.zeros((len(valid_matches), 2), dtype=np.float32)
    for i, match in enumerate(valid_matches):
        point1_coords[i] = key_points1[match.queryIdx].pt
        point2_coords[i] = key_Points2[match.trainIdx].pt
    
    # Find the best homography using a modified RANSAC algorithm
    H, mask = cv.findHomography(point2_coords, point1_coords, cv.USAC_ACCURATE, confidence=1)
    
    num_inliers = np.count_nonzero(mask)
    # print("Inliers: ", num_inliers)

    if np.count_nonzero(mask) < 1:
        mask = (np.array(mask).flatten()).astype(bool)
        img_matches = cv.drawMatches(img1, key_points1, img2, key_Points2, valid_matches[mask], None, flags=2)
        debug.showImg([img_matches], locals())

    retVal = num_inliers > 20
    return retVal, H

def padCoordinates(coords):
    corners_len = coords.shape[0]
    corners_padded = np.hstack([coords, np.ones((corners_len, 1))])
    return corners_padded

def drawCorners(img, corners):
    img_corners = img.copy()
    for corner in corners.T:
        pos=np.rint(corner).astype(int)
        cv.drawMarker(img_corners,(pos[0], pos[1]), color=(255, 255, 255), markerSize=5,thickness=2)

    return img_corners

def getWorldCoordOfChessboard():
    objp = np.zeros((3*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:3].T.reshape(-1,2)*40/15
    objp = np.array([[0, 2, 0], [1, 2, 0], [2, 2, 0], [3, 2, 0], [4, 2, 0],
                    [5, 2, 0], [6, 2, 0],
                    [0, 1, 0], [1, 1, 0], [2, 1, 0], [3, 1, 0], [4, 1, 0],
                    [5, 1, 0], [6, 1, 0],
                    [0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0],
                    [5, 0, 0], [6, 0, 0]])*40/15
    
    return objp

def siftMatching(cam, img_original, corners_original, matcherType='SIFT'):
    # Initiate detector    
    # MATCH TO ORIGINAL
    MATCH_ORIG = False

    if matcherType=='SIFT':
        print("Using SIFT feature detection")
        descriptor = cv.SIFT_create()
        matcher = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    elif matcherType=='ORB':
        # print("Using ORB detection")
        descriptor = cv.ORB_create()
        matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    img_original_rgb = cv.cvtColor(img_original, cv.COLOR_BGR2RGB)
    
    #Read camera matrix
    camera_matrix, dist_coeffs = cam.readCalibMatrix()
    translation_vector = 0
    img1 = img_original
    i = 0
    while i == 0:

        _, img2 = cam.read()
        img2_orig = img2.copy()
        # debug.showImg([img_original, img2], locals())
        
        start_time = time.time()

        # finds the Homography matrix to tranform img onto img_original        
        match_ret, H = detectAccurateMatches(img1, img2, descriptor, matcher)

        # not enough inliers found
        if not match_ret:
            continue

        Hinv = np.linalg.inv(H)

        # Padd the corners to be transformed using homograpgy matrix
        corners_original_padded = padCoordinates(corners_original)
        corners_Projected_padded = (Hinv @ corners_original_padded.T)
        corners_Projected = corners_Projected_padded[0:2, :]
        
        end_time = time.time()
        # print("Feature detection Time (s): ", end_time-start_time)

        ## Match to original image
        if MATCH_ORIG:
            match_ret, H_original = detectAccurateMatches(img_original, img2, descriptor, matcher)
            H_inv_orig = np.linalg.inv(H_original)

            corners_Projected_Orig = (H_inv_orig @ corners_original_padded.T)[0:2, :]

            img_corners_orig = drawCorners(img2_orig, corners_Projected_Orig)
            debug.showImg([img_corners_orig], locals())
        
        # Warp img onto the position of previous image
        height, width, _ = img2.shape
        img2_rot = cv.warpPerspective(img2, H , (width, height))

        # # draw lines on the warped image to compare the matches
        # cv.line(img2_rot, [0, 640], [480*2, 640], (255,255,255), thickness=2)
        # cv.line(img2_rot, [480, 0], [480, 640*2], (255,255,255), thickness=2)
        # img2_rot = cv.cvtColor(img2_rot, cv.COLOR_BGR2RGB)
        objp = getWorldCoordOfChessboard()

        success, rotation_vector, translation_vector = cv.solvePnP(
            objp,
            corners_Projected.T,
            camera_matrix,
            dist_coeffs)
        
        # print("Camera Position: ", translation_vector.ravel()*15)
        
        axis = np.float32([[1,0,0], [0,1,0], [0,0,-1]]).reshape(-1,3)*40/15
        imgpts, jac = cv.projectPoints(
            axis, rotation_vector, translation_vector,
            camera_matrix, dist_coeffs)
        
        ## print corners
        # img_corners = drawCorners(img2, corners_Projected)
        # img_corners = draw(img_corners, corners_Projected[:,14].T,imgpts)
        # debug.showImg([img_corners], locals())

        # plt.imshow(np.hstack([img1, img2_rot]))
        # plt.show()
        # img2_cropped = cropImgToBoard(img2_rot, corners_Projected.T)
        # img1 = img2_cropped.copy()
        i += 1

    board_pos = translation_vector.reshape((1,3))

    global xflip
    global yflip
    if xflip == 0:
        xflip = -1 if board_pos[0, 0] < 0 else 1
        yflip = -1 if board_pos[0, 1] > 0 else 1
        print("FLIPPPP")

    board_pos = 15 * board_pos * np.array([xflip, yflip, 1]).reshape((1,3))

    return board_pos

def draw(img, corner, imgpts):
    imgpts = imgpts.astype(int)
    pt1 = tuple(corner.astype(int))
    img = cv.line(img, pt1, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv.line(img, pt1, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, pt1, tuple(imgpts[2].ravel()), (0,0,255), 5)

    return img
def findSquareContours(img, contours, minArea, maxArea, drawCorners):
    """
    Receives list of contours and filters out squares
    To be a square a contours must:
        - Have 4 lines approximating it
        - have an area between min and max
        - have an aspect ratio between 0.8 and 1.2
    """
    squares = []
    for cnt in contours:
        x1,y1 = cnt[0][0]
        approx = cv.approxPolyDP(cnt, 0.03*cv.arcLength(cnt, True), True)
        area = cv.contourArea(cnt)
        if approx.shape[0] == 4 and area > minArea and area < maxArea:
            x, y, w, h = cv.boundingRect(cnt)
            ratio = float(w)/h
            if ratio >= 0.8 and ratio <= 1.2:
                squares.append(cnt)
                if drawCorners:
                    img = cv.drawContours(img, [cnt], -1, (0,255,255), 3)

    return img, squares

def estimate_coef(x, y):
    """
    Calculate linear regresion coef
    """
    # number of observations/points
    n = np.size(x)
  
    # mean of x and y vector
    m_x = np.mean(x)
    m_y = np.mean(y)
  
    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x
  
    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
  
    return (b_0, b_1)

def findBoardCentreSquares(img, printImgs):
    """
    Finds the centre of chessboard.
    1. applies adaptive threshold to find grayscale image
    2. dilates image to seperate different squares from each other
    3. finds square contours
    4. delete everything except square contours
    5. erode image to return squares to original size
    6. use findChessboardcorner function

    """
    patternSize = (7, 3)

    # convert image to a grayscale image
    img = cv.GaussianBlur(img, (5,5), 0.5)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # threshhold it into a binary image
    # a large window size is used 71 to only change
    # threshold slowly
    img_adpt_thresh = cv.adaptiveThreshold(
        img_gray, 
        255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY_INV,
        71,
        2
    )

    # dilate the image to grow the white areas
    # leaving small black squares in the centre of open squares
    kernSize = 7
    img_dilate = cv.dilate(img_adpt_thresh, np.ones((kernSize, kernSize)))

    # find the points of contours aproximate contours
    contours, hierarchy = cv.findContours(img_dilate, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # check each contours and save squares
    img_gray, squares = findSquareContours(img_gray, contours, minArea=100, maxArea=6000, drawCorners=printImgs)

    # Create fully white image and fill poly with black.
    # this leaves only the black squares of the chessboard in the image
    img_mask = np.ones((480, 640), dtype=np.uint8)*255
    cv.fillPoly(img_mask, squares, 0)

    #Dilate the area to return the squares to their original size
    img_mask = cv.erode(img_mask, np.ones((kernSize-1, kernSize-1)))
    retVal, corners = cv.findChessboardCorners(img_mask, patternSize)
    print("Board found: ", retVal)

    debug.saveTempImg(img_mask, "board_mask.png")
    
    # Debug lines to plot different image stages
    if printImgs:
        cv.drawChessboardCorners(img_gray, patternSize, corners, retVal)
        debug.showImg([img_gray, img_adpt_thresh, img_dilate, img_mask], locals())

    return retVal, corners

def getRotationMatrix(corners):
    """
    Finds the rotation matrix using the interior corners of the chessboard.
    The rotation matrix rotates the image around the center of the board.
    To align the board with the horizontal axis.

    
    Find the transformation matrix to transform the image frame to the base
    co-ordinate frame.

    The transformation moves the centerpoint to the origin and flips the y axis.
    """
    # calculate the linear regression of each row to find the 
    # rotation of the board.
    rows = np.zeros(shape=(3, 7, 2))
    coef = np.zeros(shape=(3,2))
    for i in range(3):
        rows[i] = np.reshape(corners[7*i:7*i+7, :, :], (7, 2))
        coef[i] = estimate_coef(rows[i, :, 0], rows[i, :, 1])
    
    # the orgin is the centre corner point of the chessboard.
    # the rotation angle is in degrees and positive CCW.
    gradient = np.mean(coef[:, 1])
    angle= np.arctan2(gradient, 1)
    origin = corners[10, 0, :]

    # Find a rotation matrix to rotate the image around the origin.
    # it also flips the image along the vertical axis
    # b0 = (1-np.cos(angle))*origin[0]-np.sin(angle)*origin[1]-origin[0]
    # b1 = np.sin(angle)*origin[0]+(1-np.cos(angle))*origin[1]-origin[1]

    R = np.array([
        [np.cos(angle), np.sin(angle)],
        [np.sin(angle), -np.cos(angle)]
    ])

    # T = np.reshape(-(R@origin), (2,1))
    T = np.reshape(origin, (2,1))

    # affine tranformation for image transformation
    M = cv.getRotationMatrix2D(origin, angle*180/np.pi, 1)
    # move origin to zero
    # M[:, 2] -= origin
    # flip y axis
    # M[1, :] = M[1, :]*-1 

    return R, T, M

def drawImgInBaseCoords(img, rotationMatrix):
    (h, w) = img.shape[:2]
    # imgOffset = np.array([[0, 0, w], [0, 0, h]])
    imgRot = cv.warpAffine(img, rotationMatrix, dsize=(w, h))
    imgDrawn = imgRot.copy()
    # draw lines on axis of image
    cv.line(imgDrawn, [0, 640], [480*2, 640], (255,255,255), thickness=2)
    cv.line(imgDrawn, [480, 0], [480, 640*2], (255,255,255), thickness=2)

    return imgRot, imgDrawn

def processInitialImg(img):
    retVal, corners = findBoardCentreSquares(img, printImgs=True)

    # raise exception when board not found
    if not retVal:
        cv.waitKey(0)
        raise("Could not find the chessboard pattern within the starting position \
              chess Board. Check that the board is visible.")

    # Find rotation matrix to align image with axes. 
    # assume centerpoint is originqq
    R, Origin, rotationMatrix = getRotationMatrix(corners)

    imgRot, imgDrawn = drawImgInBaseCoords(img, rotationMatrix)

    corners1 = np.reshape(corners, (21, 2))

    cornersTransformed = (R @ (corners1.T - Origin)).T + Origin.T

    return imgRot, cornersTransformed, Origin

def cropImgToBoard(img_orig, corners):
    """
    Crops the image to slightly larger than the chessboard.
    Does not change the image shize. Only changes outside pixels
    to black.
    """
    img = img_orig.copy()
    height, width, _ = img.shape
    min_h, max_h = np.min(corners[:, 0]), np.max(corners[:, 0])
    min_y, max_y =  np.min(corners[:, 1]), np.max(corners[:, 1])

    d = 40
    y_diff = np.diff([corners[0:7,0], corners[7:14,0], corners[14:,0]])
    y_diff = np.mean(y_diff).astype(int)

    min_h = np.clip(min_h-y_diff-d, 0, width).astype(int)
    max_h = np.clip(max_h+y_diff+d, 0, width).astype(int)
    min_y = np.clip(min_y-y_diff*3 -d, 0, height).astype(int)
    max_y = np.clip(max_y+y_diff*3+d, 0, height).astype(int)

    print(min_h, max_h)
    print(min_y, max_y)

    img[: , 0:min_h, :] = 0
    img[: , max_h:, :] = 0

    img[0:min_y,:, :] = 0
    img[max_y: ,:, :] = 0

    return img


boardOffset = np.array([-120, 200, 0]).reshape((-1, 1));
def convertBoardToRobotCoords(boardCoords):
    return boardCoords + boardOffset

def convertRobottoBoardCoords(robotCoords):
    return robotCoords - boardOffset

def getCam():
    if platform.system() == "Windows":
        imgPath = Path("Chessboard_detection", "TestImages", "22_04_2023", "1")
        cam = Camera_Manager.FakeCamera((480, 640), str(imgPath.resolve()))
    elif platform.system() == "Linux":
        imgPath = Path("Chessboard_detection", "TestImages", "Temp")
        cam = Camera_Manager.RPiCamera((480, 640), imgPath.resolve(), storeImgHist=False, loadSavedFirst=False)
    else:
        raise("UNKNOWN OPERATING SYSTEM TYPE")
    
    return cam

def get_input(queue):
    while True:
        user_input = sys.stdin.readline().rstrip('\n')
        queue.put(user_input)

def string_to_array(s: str) -> np.ndarray:
    # Split the string into a list of strings
    str_list = s.split(',')
    
    # Convert the strings to integers
    int_list = [int(s.strip()) for s in str_list]
    
    # Convert the list of integers to a NumPy array
    int_array = np.array(int_list)
    
    return int_array

def main():
    cam = getCam()

    # intialize robot pos
    cm = ChessMoves(lift=200)
    mc = MotorCommands()

    start_pos_robot = np.array([0, 230, 500]).reshape((-1, 1))
    target_pos_board = np.array([150, -50, 400]).reshape((-1, 1))
    target_robot_pos = convertBoardToRobotCoords(target_pos_board)

    # save initial position
    _, img = cam.read()
    debug.saveTempImg(img, "Start_Pos.png")

    # go to start position
    angles = cm.inverse_kinematics(start_pos_robot)
    mc.go_to(mc.sort_commands(angles, 0))

    time.sleep(5)
    # Find the chessboard center and rotate the image to lie on the center
    imgRotated, cornersOrigin, BoardOrigin = processInitialImg(img)
    # imgRotated = cropImgToBoard(imgRotated, cornersOrigin)

    # input_queue = queue.Queue()
    # input_thread = threading.Thread(target=get_input, args=(input_queue,))
    # input_thread.daemon = True
    # input_thread.start()

    current_board_pos = convertRobottoBoardCoords(start_pos_robot)
    error_temp = np.array([0])
    print("detected_pos, \t error, \t current_target")
    while 1:
        if (error_temp == 0).all():
            value = input("Enter new target:")
            print("Target Pos: ", value)
            target_pos_board = string_to_array(value).reshape((-1, 1))

        detected_board_pos = siftMatching(cam, imgRotated, cornersOrigin, matcherType='ORB')
        # print("Detected pos: ", board_pos)

        error = target_pos_board - detected_board_pos.reshape((-1, 1))
        error_temp = error.copy()
        error_temp[np.abs(error_temp)<8] = 0
        error_temp = np.clip(error_temp, -20, 20)

        # print("error: ", error.ravel())
        print_detected_pos = detected_board_pos.ravel()
        print_error = error.ravel()
        print_current_pos = current_board_pos.ravel()
        # print(print_detected_pos, print_error, print_current_pos)
        print(print_detected_pos)

        current_board_pos = current_board_pos + error_temp*0.5
        robot_coords = convertBoardToRobotCoords(current_board_pos)
        # angles = cm.inverse_kinematics(robot_coords)
        # mc.go_to(mc.sort_commands(angles, 0))
        
        # time.sleep(2)
        
        
    return 0

if __name__ == "__main__":
    main()