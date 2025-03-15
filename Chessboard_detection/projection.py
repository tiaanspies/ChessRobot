import cv2 as cv
from time import sleep
from Chessboard_detection import board_detection_hough as bdh
import numpy as np

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
    IMG_SIZE = 304
    corners_src_outer = np.array([corners_src[0, 0], corners_src[0, -1], corners_src[-1, 0], corners_src[-1, -1]], dtype=np.float32)
    corners_dest = np.array([[0,0], [IMG_SIZE, 0], [0, IMG_SIZE], [IMG_SIZE, IMG_SIZE]], dtype=np.float32)

    M = cv.getPerspectiveTransform(corners_src_outer, corners_dest)
    img_warped = cv.warpPerspective(img, M, (IMG_SIZE, IMG_SIZE))

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

def process_image_single_board(img):
    """
    Pipeline to find chessboard, split into squares and crop away the border.

    Parameters:
    img: Image of the chessboard pattern.

    Returns:
    cropped_image: Image of the chessboard pattern cropped to only contain the board.
    """

    corners = bdh.find_board_corners(img)
    # find offsets to overlay img 1 and 2 perfectly
    cropped_image = project_board_2_square(img, corners) 

    return cropped_image

def process_image_multiple_squares(img):
    """
    Pipeline to find chessboard, split into squares and crop away the border.

    Parameters:
    img: Image of the chessboard pattern.

    Returns:
    List of 64 square images of the chessboard pattern.
    """

    corners = bdh.find_board_corners(img)
    # find offsets to overlay img 1 and 2 perfectly
    cropped_image = project_board_2_square(img, corners) 

    # split the board int
    squares = split_image_into_squares(cropped_image)

    # shrink the squares to remove the border
    squares = crop_squared_by_border_width(squares, border_width=2)

    return squares