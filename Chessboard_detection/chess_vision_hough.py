import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import MeanShift, estimate_bandwidth

import Chessboard_detection.board_detection_hough as bdh
import Chessboard_detection.projection as proj



class ChessVisionHough:
    def __init__(self, img):
        # Process image into individual squares snippets (Array of 64 squares each 2D numpy array with 3 BGR channels)
        squares = proj.process_image(img)

        # preprocess the squares (apply Gaussian blur and convert to HSV)
        blurred_hsv_squares = [preprocess_square(square) for square in squares]

        # get the linear regression model to determine if a square has a piece or not.
        self.model_piece_vs_empty = self.get_piece_vs_empty_model(blurred_hsv_squares)

        # w_o_w, w_o_b, b_o_w, b_o_b, e_w, e_b = get_init_board_groups()

    def get_piece_vs_empty_model(self, squares):
        """
        Inputs a list of 64 squares and returns a logistic regression model that determines if a square has a piece or not.
        """
        features = extract_features(squares)
        labels = get_init_board_labels_piece_v_empty()
        model = get_logistic_regression_model(features, labels)

        # plot_decision_boundary(model, features, labels)

        return model

    def indentify_piece_ids(self, img):
        """
        Directly find chessboard corners in img2 and warps it to square and crops it.
        """
        squares = proj.process_image(img)

        # preprocess the squares (apply Gaussian blur and convert to HSV)
        blurred_hsv_squares = [preprocess_square(square) for square in squares]

        piece_v_empty = self.model_piece_vs_empty.predict(extract_features(blurred_hsv_squares))

        for i in range(64):
            if piece_v_empty[i] == 1:
                piece_v_empty[i] = determine_piece_color(squares[i])
        return [-1] * 64
    
    def label_chessboard(self, img):
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
        label_ids = self.indentify_piece_ids(img)
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
    
def determine_piece_color(square):
    """
    Use canny edge detection to find the outline of the piece in the square.
    Then use the color of the pixels inside the outline to determine the color of the piece.
    """
    # # Convert the image to the LUV color space
    luv_square = cv2.cvtColor(square, cv2.COLOR_BGR2LUV)

    # # Apply mean shift segmentation
    # shifted = cv2.pyrMeanShiftFiltering(luv_square, sp=11, sr=11)

    # # Convert back to BGR color space
    # output = cv2.cvtColor(shifted, cv2.COLOR_LUV2BGR)
    # proj.showImg(output)

    # blur to reduce noise
    img = cv2.medianBlur(square, 3)

    # flatten the image
    luv_square = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    flat_image = luv_square.reshape((-1,3))
    flat_image = np.float32(flat_image)

    # meanshift
    bandwidth = estimate_bandwidth(flat_image, quantile=.2)
    ms = MeanShift(bandwidth=bandwidth, max_iter=800, bin_seeding=True)
    ms.fit(flat_image)
    labeled=ms.labels_


    # get number of segments
    segments = np.unique(labeled)
    print('Number of segments: ', segments.shape[0])

    #create a mask of the boder pixels and flatten it
    mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)
    mask[1:-1, 1:-1] = 0
    mask = mask.reshape((-1,1))

    # count the number of border pixels in each segment
    border_count = np.zeros(segments.shape[0])
    for i, label in enumerate(labeled):
        border_count[label] += mask[i]

    # for squares with 2 or 3 segments, remove the pixels of the segment with the most border pixels.
    # for squares for more than 3 segments. Removes pixels of border segments with more 30% of the border pixels.

    for i, label in enumerate(labeled):
        if segments.shape[0] == 2 or segments.shape[0] == 3:
            if border_count[label] == max(border_count):
                labeled[i] = -1
        elif segments.shape[0] > 3:
            if border_count[label] > 0.3 * sum(border_count):
                labeled[i] = -1
    # print the border count as a ratio of the whole border
    border_ratio = border_count / sum(border_count)
    print('Border count ratio: ', border_ratio)

    # get the average color of each segment
    total = np.zeros((segments.shape[0], 3), dtype=float)
    count = np.zeros(total.shape, dtype=float)
    for i, label in enumerate(labeled):
        if label == -1:
            continue
        total[label] = total[label] + flat_image[i]
        count[label] += 1
        
    avg = total/count
    avg = np.uint8(avg)

    # cast the labeled image into the corresponding average color
    green = np.array([125, 100, 50], dtype=np.uint8)
    res = np.array([avg[label] if label != -1 else green for label in labeled])
    result = res.reshape((img.shape))
    result = cv2.cvtColor(result, cv2.COLOR_LUV2BGR)


    proj.showImg(square)
    proj.showImg(result)

    return 1

def extract_features(squares):
    features = []
    for square in squares:
        mean_s, std_s = cv2.meanStdDev(square[:,:,1])
        mean_v, std_v = cv2.meanStdDev(square[:,:,2])

        features.append([std_s[0,0], std_v[0,0]])

    return np.array(features)

def get_logistic_regression_model(features, labels):
    """
    Inputs a preprocessed HSV image of the sqyare.

    Fits a Gaussian curve to the histogram of the S and V channels to determine if there is a empty square or not.

    Empty squares will have a single prominent peak in the S and V channels where with a piece it will be bimodal or more spread out.
    """

    model = LogisticRegression()
    model.fit(features, labels)

    return model

def get_init_board_labels_piece_v_empty():
    """
    
    Return an array [64] where 1 is a piece and 0 is empty.
    """

    labels = [1]*16+[0]*32+[1]*16

    return labels

def preprocess_square(square):
    blurred = cv2.GaussianBlur(square, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    return hsv

def plot_decision_boundary(model, features, labels):
    """
    Plots the decision boundary of the logistic regression model along with the data points.
    """
    x_min, x_max = features[:, 0].min() - 1, features[:, 0].max() + 1
    y_min, y_max = features[:, 1].min() - 1, features[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(features[:, 0], features[:, 1], c=labels, edgecolors='k', marker='o')
    plt.xlabel('Standard Deviation of S Channel')
    plt.ylabel('Standard Deviation of V Channel')
    plt.title('Logistic Regression Decision Boundary')
    plt.show()

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
    

def label_dict_to_str(label_dict):
    """
    Inputs a dict label_dict[SQUARES[i]] = {"piece": labels_words[i], "background": square_colour[i]}

    outputs a dict label_dict_words[SQUARES[i]] = f"{labels_words[i]} on {square_colour[i]}"
    """

    label_dict_words = {}

    for square, labels in label_dict.items():
        label_dict_words[square] = f"{labels['piece']} on {labels['background']}"

    return label_dict_words


def get_init_board_groups():
    """
    Returns the IDS for the initial board configuration. The 6 groups are white pieces on white squares, white pieces on black squares, 
    black pieces on white squares, black pieces on black squares, empty white squares and empty black squares.

    returns:
    white_on_white: array[8] of ids.
    white_on_black: array[8] of ids.
    black_on_white: array[8] of ids.
    black_on_black: array[8] of ids.
    empty_white: array[16] of ids.
    empty_black: array[16] of ids.
    """

    white_on_white = [1,3,5,7,8,10,12,14]
    white_on_black = [0,2,4,6,9,11,13,15]
    empty_white = [17,19,21,23,24,26,28,30,33,35,37,39,40,42,44,46]
    black_on_white = [48,50,52,54,57,59,61,63]
    black_on_black = [49,51,53,55,56,58,60,62]
    empty_black = [16,18,20,22,25,27,29,31,32,34,36,38,41,43,45,47]

    return white_on_white, white_on_black, black_on_white, black_on_black, empty_white, empty_black

def display_squares(squares, contrast_squares, preprocessed_squares, has_pieces):
    """
    Display 3 rows of 8 squares each. Each square is displayed with the original, preprocessed and has_piece image.
        
    """
    for i in range(8):
        plt.figure(figsize=(17, 9))
        for j in range(8):
            plt.subplot(4, 8, 1 + j)
            plt.title('Original Image')
            plt.imshow(cv2.cvtColor(squares[i*8+j],cv2.COLOR_BGR2RGB))

            # plt.subplot(4, 8, 9 + j)
            # plt.title('Enhanced Contrast')
            # plt.imshow(contrast_squares[i*8+j], cmap='gray')

            # mean_s, std_s = cv2.meanStdDev(preprocessed_squares[i*8+j][:,:,1])
            # mean_v, std_v = cv2.meanStdDev(preprocessed_squares[i*8+j][:,:,2])            

            # plt.subplot(4, 8, 17 + j)
            # plt.title('Histogram (S) P:'+str(has_pieces[i*8+j]))
            # plt.hist(preprocessed_squares[i*8+j][:,:,1].ravel(), bins=256, range=[0, 256], color='black')

            # plt.subplot(4, 8, 25 + j)
            # plt.title(str((std_s)) + ' ' + str(std_v))
            # plt.hist(preprocessed_squares[i*8+j][:,:,2].ravel(), bins=256, range=[0, 256], color='black')
    
        plt.show()


def display_group(group1, group2, group3, group4, group_1_name, group_2_name, group_3_name, group_4_name):
    """
    Display 4 rows of 8 squares each. Each square is displayed with the original, preprocessed and has_piece image.
        
    """
    for i in range(len(group1)//8):
        plt.figure(figsize=(17, 9))
        for j in range(8):

            hsv_img1 = cv2.cvtColor(group1[i*8+j], cv2.COLOR_BGR2HSV)
            hsv_img2 = cv2.cvtColor(group2[i*8+j], cv2.COLOR_BGR2HSV)
            hsv_img3 = cv2.cvtColor(group3[i*8+j], cv2.COLOR_BGR2HSV)
            hsv_img4 = cv2.cvtColor(group4[i*8+j], cv2.COLOR_BGR2HSV)

            plt.subplot(4, 8, j + 1)
            plt.title('H Channel')
            plt.hist(hsv_img1[:, :, 0].ravel(), bins=180, range=[0, 180], color='blue', alpha=0.5, label=group_1_name)
            plt.hist(hsv_img2[:, :, 0].ravel(), bins=180, range=[0, 180], color='red', alpha=0.5, label=group_2_name)
            plt.hist(hsv_img3[:, :, 0].ravel(), bins=180, range=[0, 180], color='green', alpha=0.5, label=group_3_name)
            plt.hist(hsv_img4[:, :, 0].ravel(), bins=180, range=[0, 180], color='orange', alpha=0.5, label=group_4_name)
            plt.legend()

            plt.subplot(4, 8, 9 + j)
            plt.title('S Channel')
            plt.hist(hsv_img1[:, :, 1].ravel(), bins=256, range=[0, 256], color='blue', alpha=0.5, label=group_1_name)
            plt.hist(hsv_img2[:, :, 1].ravel(), bins=256, range=[0, 256], color='red', alpha=0.5, label=group_2_name)
            plt.hist(hsv_img3[:, :, 1].ravel(), bins=256, range=[0, 256], color='green', alpha=0.5, label=group_3_name)
            plt.hist(hsv_img4[:, :, 1].ravel(), bins=256, range=[0, 256], color='orange', alpha=0.5, label=group_4_name)
            plt.legend()

            plt.subplot(4, 8, 17 + j)
            plt.title('V Channel')
            plt.hist(hsv_img1[:, :, 2].ravel(), bins=256, range=[0, 256], color='blue', alpha=0.5, label=group_1_name)
            plt.hist(hsv_img2[:, :, 2].ravel(), bins=256, range=[0, 256], color='red', alpha=0.5, label=group_2_name)
            plt.hist(hsv_img3[:, :, 2].ravel(), bins=256, range=[0, 256], color='green', alpha=0.5, label=group_3_name)
            plt.hist(hsv_img4[:, :, 2].ravel(), bins=256, range=[0, 256], color='orange', alpha=0.5, label=group_4_name)
            plt.legend()
    
        plt.show()

    
def main():
    img_path2 = str(Path("Chessboard_detection", "TestImages", "Temp", "a2.jpg"))
    img_starting_pos = cv2.imread(img_path2)

    chess_vision = ChessVisionHough(img_starting_pos)

    chess_vision.label_chessboard(img_starting_pos)

if __name__ == "__main__":
    main()