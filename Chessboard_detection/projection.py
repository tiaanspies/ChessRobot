import random
import cv2 as cv
from pathlib import Path
from time import sleep
from Chessboard_detection import board_detection_hough as bdh
import numpy as np
import heapq
import matplotlib.pyplot as plt
import chess

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
    
    White has to be at the bottom row of the image.

    Returns 64 squares of the chessboard pattern. Each square is a 2D numpy array.
    """
    
    # split the image into 8x8 squares
    height, width = img.shape[:2]
    squares = []
    square_height = height // 8
    square_width = width // 8
    for i in range(8):
        for j in range(8):
            y = (7 - i) * square_height
            x = j * square_width
            square = img[y:y+square_height, x:x+square_width]
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
    blurred = cv2.GaussianBlur(square, (5, 5), 0)
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

def display_squares(squares, contrast_squares, preprocessed_squares, has_pieces):
    """
    Display 3 rows of 8 squares each. Each square is displayed with the original, preprocessed and has_piece image.
        
    """
    for i in range(8):
        plt.figure(figsize=(17, 9))
        for j in range(8):
            plt.subplot(4, 8, 1 + j)
            plt.title('Original Image')
            plt.imshow(squares[i*8+j], cmap='gray')

            plt.subplot(4, 8, 9 + j)
            plt.title('Enhanced Contrast')
            plt.imshow(contrast_squares[i*8+j], cmap='gray')

            plt.subplot(4, 8, 17 + j)
            plt.title('Histogram (Original)')
            plt.hist(squares[i*8+j].ravel(), bins=256, range=[0, 256], color='black')

            plt.subplot(4, 8, 25 + j)
            plt.title('Histogram (Enhanced)')
            plt.hist(contrast_squares[i*8+j].ravel(), bins=256, range=[0, 256], color='black')

            # plt.subplot(4, 8, 17 + j)
            # plt.title('Thresholded')
            # plt.imshow(preprocessed_squares[i*8+j], cmap='gray')

            # plt.subplot(4, 8, 25 + j)
            # plt.title('Has piece: ' + str(has_pieces[j][0]))
            # plt.imshow(has_pieces[i*8+j][1], cmap='gray')
    
        plt.show()

def chessboard_detection_3_hough(img):
    """
    Directly find chessboard corners in img2 and warps it to square and crops it.
    """
    corners = bdh.find_board_corners(img)
    # find offsets to overlay img 1 and 2 perfectly
    cropped_image = project_board_2_square(img, corners) 
    
    board_colors = board_background_labels()
    # TODO: Pick up coding here 

    # split the board int
    squares = split_image_into_squares(cropped_image)

    # shrink the squares to remove the border
    squares = crop_squared_by_border_width(squares, border_width=1)

    # enhance the contrast of the squares
    gray_squares = [cv2.cvtColor(square, cv2.COLOR_BGR2GRAY) for square in squares]
    contrast = [cv2.equalizeHist(square) for square in gray_squares]

    # preprocess the squares
    preprocessed_squares = [preprocess_square(square) for square in contrast]

    # check if the square has a piece
    has_pieces = [has_piece(square) for square in preprocessed_squares]

    # Display the results
    # display_squares(squares, contrast, preprocessed_squares, has_pieces)

    return [-1] * 64

def convert_ids_to_words(labels):
    """
    Coverts IDS 1, 0, -1 to correspnding colors white, none and black.

    Parameters:
    labels: array[64] of ids.
    Returns:
    array[64] of strings.
    """

    mapping = ["black", "none", "white"]
    labels = [mapping[id+1] for id in labels]
    
    return labels

def board_background_labels():
    """
    Returns the labels for the background of the chessboard.

    returns:
    array[64] of strings.
    """

    labels = []
    for i in range(8):
        for j in range(8):
            if (i+j) % 2 == 0:
                labels.append("black")
            else:
                labels.append("white")
                
    return labels
    

def label_chessboard(img):
    """
    Labels the chessboard squares with detected pieces and their background colors.
    This function takes an image of a chessboard, detects the pieces on each square,
    and assigns a label to each square indicating the piece and the background color.
    Args:
        img (numpy.ndarray): The input image of the chessboard.
    Returns:
        dict: A dictionary where each key is a square identifier (e.g., "A1", "B1", etc.)
              and the value is another dictionary with keys "piece" and "background".
              The "piece" key maps to the detected piece on that square (as a string),
              and the "background" key maps to the background color of the square.
    Example:
        >>> img = cv2.imread('chessboard.jpg')
        >>> labels = label_chessboard(img)
        >>> print(labels["A1"])
        {'piece': 'white', 'background': 'white'}
    """
    label_ids = chessboard_detection_3_hough(img)
    labels_words = convert_ids_to_words(label_ids)

    square_colour = board_background_labels()

    SQUARES = [
        "a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1",
        "a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2",
        "a3", "b3", "c3", "d3", "e3", "f3", "g3", "h3",
        "a4", "b4", "c4", "d4", "e4", "f4", "g4", "h4",
        "a5", "b5", "c5", "d5", "e5", "f5", "g5", "h5",
        "a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6",
        "a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7",
        "a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8",
    ]

    label_dict = {}

    for i in range(64):
        label_dict[SQUARES[i]] = {"piece": labels_words[i], "background": square_colour[i]}

    return label_dict

def label_dict_to_str(label_dict):
    """
    Inputs a dict label_dict[SQUARES[i]] = {"piece": labels_words[i], "background": square_colour[i]}

    outputs a dict label_dict_words[SQUARES[i]] = f"{labels_words[i]} on {square_colour[i]}"
    """

    label_dict_words = {}

    for square, labels in label_dict.items():
        label_dict_words[square] = f"{labels['piece']} on {labels['background']}"

    return label_dict_words

def main():
    img1 = str(Path("Chessboard_detection", "TestImages", "Temp", "1.jpg"))
    blank_board = cv2.imread(img1)

    img_path2 = str(Path("Chessboard_detection", "TestImages", "Temp", "a2.jpg"))
    img2 = cv2.imread(img_path2)

    # find_warped_2(img1, img2)
    chessboard_detection_3_hough(blank_board, img2)

if __name__ == "__main__":
    main()