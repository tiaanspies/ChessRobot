import random
import cv2 as cv
from pathlib import Path
from time import sleep
from Chessboard_detection import board_detection_hough as bdh
import numpy as np
import heapq
import matplotlib.pyplot as plt

"""
Manages image projection and perspective transformations so that the chessboard pattern standardized in the image.

"""

import cv2

def project_board_2_square(img, corners_src):
    """
    Input the image and the corners of the chessboard. 

    Input image contains a chessboard pattern but from the camera angle it is not square in the image. 
    This function will project it to be square and then crop the rest of the image. Leacinf only a sqaure chessboard.

    returns an image that is filled with the chessboard pattern and is square in the image.
    """
    corners_src_outer = np.array([corners_src[0, 0], corners_src[0, -1], corners_src[-1, 0], corners_src[-1, -1]], dtype=np.float32)
    corners_dest = np.array([[0,0], [300, 0], [0, 300], [300, 300]], dtype=np.float32)

    M = cv.getPerspectiveTransform(corners_src_outer, corners_dest)
    img_warped = cv.warpPerspective(img, M, (300, 300))

    return img_warped

def split_image_into_squares(img):
    """
    Inputs a topdown image of the chessboard pattern. Image must be cropped to only contain the board.
    
    Returns 64 squares of the chessboard pattern. Each square is a 2D numpy array.
    """
    
    # split the image into 8x8 squares

    width, height = img.shape[:2]
    squares = []
    for i in range(8):
        for j in range(8):
            x = j * width // 8
            y = i * height // 8
            square = img[y:y+height//8, x:x+width//8]
            squares.append(square)
    
    return squares
 
def showImg(img):
    # while cv.waitKey(1) != ord('q'):
    # for i, img in enumerate(images):
    cv.namedWindow("1", cv.WINDOW_NORMAL)
    
    height, width = img.shape[:2]
    cv.resizeWindow("1", width, height)
    cv.imshow("1", img)

    while cv.waitKey(1) != ord('q'):
        sleep(0.1)

def preprocess_square(square):
    gray = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_OTSU)
    return thresh

def has_piece(thresh, min_contour_area=100):
    """
    Detects contours in a binary image (thresh) and returns True if the contour area is greater than min_contour_area.

    Parameters:
    thresh: Binary image.
    min_contour_area: Minimum contour area to be considered as a piece.
    returns:
    True if the contour area is greater than min_contour_area
    """
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv.contourArea(cnt)
    
    color_thresh = cv.cvtColor(thresh, cv.COLOR_GRAY2RGB)
    cv.drawContours(color_thresh, contours, -1, (0, 0, 255), 2)

    
    return len(contours) > 0, color_thresh

def crop_squared_by_border_width(squares, border_width=2):
    """
    Crops the square by the border width to remove the border of the square.
    Parameters:
    squares: List of 64 squares of the chessboard pattern. Each square is a 2D numpy array.
    border_width: The width of the border to crop from the square.
    Returns:
    List of 64 squares of the chessboard pattern. Each square is a 2D numpy array.
    """

    cropped_squares = []
    for square in squares:
        cropped_square = square[border_width:-border_width, border_width:-border_width]
        cropped_squares.append(cropped_square)
    return cropped_squares

def crop_outer_corners(img, corners):
    """
    Inputs a topdown image of the chessboard. Crops using previously found corners to only contain the board.

    Returns a cropped image of the chessboard pattern.
    """
    corners_src_outer = np.array([corners[0, 0], corners[0, -1], corners[-1, 0], corners[-1, -1]])
    
    minX = np.min(corners_src_outer[:, 0])
    minY = np.min(corners_src_outer[:, 1])
    
    maxX = np.max(corners_src_outer[:, 0])
    maxY = np.max(corners_src_outer[:, 1])

    pattern = img[minY:maxY, minX:maxX] 

    return pattern

def find_template_location(img1, template):
    """
    Scan template across img1 to find the matching location for the pattern.
    
    Parameters:
    img1: Full size image.
    template: Smaller pattern image.
    
    Returns:
    Top-left corner of the matching location.
    """
    w, h = template.shape[:2]

    result = cv.matchTemplate(img1, template, cv.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
 
    return max_loc

def chessboard_detection_1_template_matching(img1, img2):
    """
    Uses template matching to to match the location of the chessboard in img2 to img1. Then warps and crops img2 using corners from img1.

    Parameters:
    img1: Full size image.
    img2: Full size image.

    """
    all_corners = bdh.find_board_corners(img1)

    # find offsets to overlay img 1 and 2 perfectly
    cropped_image = crop_outer_corners(img2, all_corners)    
    showImg(cropped_image)
    top_left = find_template_location(img1, cropped_image)

    # Replace the location at max_loc in img1 with the cropped image pattern
    bottom_right = (top_left[0] + cropped_image.shape[1], top_left[1] + cropped_image.shape[0])

    # Ensure the replacement area is within the bounds of img1
    if bottom_right[0] <= img1.shape[1] and bottom_right[1] <= img1.shape[0]:
        img1[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = cropped_image
        showImg(img1)
    else:
        print("Replacement area is out of bounds.")

    img_warped = project_board_2_square(img1, all_corners)
    showImg(img_warped)

def chessboard_detection_2_orb_transform(img1, img2):
    """
    Find the homographic transform to match img2 onto img1 using sift transform or similar.

    Then project the chessboard pattern in img2 to be square in the image using detected corners from img1.
    """

    # find all the corner location in img1
    all_corners = bdh.find_board_corners(img1)

    # Find and match featuers between img1 and img2
    descriptor = cv.ORB_create()
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    kp1, des1 = descriptor.detectAndCompute(img1,None)
    kp2, des2 = descriptor.detectAndCompute(img2,None)
    
    # Match descriptors.
    matches = bf.match(des1,des2)
    valid_matches = np.array(heapq.nsmallest(100, matches, key = lambda x: x.distance))

    # Get the co-ordinated of the selected matches
    point1_coords = np.zeros((len(valid_matches), 2), dtype=np.float32)
    point2_coords = np.zeros((len(valid_matches), 2), dtype=np.float32)
    for i, match in enumerate(valid_matches):
        point1_coords[i] = kp1[match.queryIdx].pt
        point2_coords[i] = kp2[match.trainIdx].pt

    # Find the best homography using a modified RANSAC algorithm
    M, mask = cv.findHomography(point2_coords, point1_coords, cv.USAC_ACCURATE, confidence=1)

    # Warp img2 to img1
    img2_warped_to_img1 = cv.warpPerspective(img2, M, (img1.shape[1], img1.shape[0]))

    img2_warped_to_img1 = project_board_2_square(img2_warped_to_img1, all_corners)

def chessboard_detection_3_hough(img1, img2):
    """
    Directly find chessboard corners in img2 and warps it to square and crops it.
    """
    corners = bdh.find_board_corners(img2)
    # find offsets to overlay img 1 and 2 perfectly
    cropped_image = project_board_2_square(img2, corners) 
    showImg(cropped_image)

    # split the board int
    squares = split_image_into_squares(cropped_image)

    # preprocess the squares
    preprocessed_squares = [preprocess_square(square) for square in squares]

    # check if the square has a piece
    has_pieces = [has_piece(square) for square in preprocessed_squares]

    # Display the results

    for square, preprocessed_square, (present, contour) in zip(squares, preprocessed_squares, has_pieces):
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 4, 1)
        plt.title('Original Image')
        plt.imshow(square, cmap='gray')

        plt.subplot(2, 4, 2)
        plt.title('Canny Edges')
        plt.imshow(preprocessed_square, cmap='gray')

        plt.subplot(2, 4, 3)
        plt.title('Has Piece: ' + str(present))
        plt.imshow(contour, cmap='gray')

        plt.show()

def main():
    img_path1 = str(Path("Chessboard_detection", "TestImages", "Temp", "1.jpg"))
    img1 = cv2.imread(img_path1)

    img_path2 = str(Path("Chessboard_detection", "TestImages", "Temp", "2.jpg"))
    img2 = cv2.imread(img_path2)

    # find_warped_2(img1, img2)
    chessboard_detection_3_hough(img1, img2)

if __name__ == "__main__":
    main()