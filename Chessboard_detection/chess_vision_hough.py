import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.mixture import GaussianMixture

import Chessboard_detection.projection as proj
import Chessboard_detection.pi_debugging as debug
import Chessboard_detection.Chess_Vision_kmeans as kmeans

class ChessVisionHough:
    def __init__(self, piece_detection_algo):
        """
        Sets up the ChessVisionHough object with the image and the piece detection algorithm.

        Parameters:
        img: 3D numpy array of the image.
        piece_detection_algo: string of the piece detection algorithm to use. ["standard", "red_green", "kmeans"]
        """
        # Process image into individual squares snippets (Array of 64 squares each 2D numpy array with 3 BGR channels)
        self.piece_detection_algo = piece_detection_algo

        

        self.square_background_means = None
        self.model_piece_vs_empty = None
        self.model_piece_color = None

        # in order to setup the class the following functions must be run
        # 1. setup_empty_board
        # 2. setup_starting_position_board
    
    def setup_empty_board(self, img):
        """
        Setup the empty board by getting the background means of the empty squares.
        """
        self.square_background_means = self.get_background_means(img)

    def setup_starting_position_board(self, img):
        """
        Setup the starting position board by getting the piece color model.
        """

        squares = proj.process_image_multiple_squares(img)

        # preprocess the squares (apply Gaussian blur and convert to HSV)
        blurred_hsv_squares = [preprocess_square(square) for square in squares]

        # get the linear regression model to determine if a square has a piece or not.
        self.model_piece_vs_empty = self.get_piece_vs_empty_model(blurred_hsv_squares)
        self.model_piece_color = self.get_piece_color_model(squares)


    def get_piece_vs_empty_model(self, squares):
        """
        Inputs a list of 64 squares and returns a logistic regression model that determines if a square has a piece or not.
        """
        features = extract_features(squares)
        labels = get_init_board_labels_piece_v_empty()
        model = get_logistic_regression_model(features, labels)

        # plot_decision_boundary(model, features, labels)

        return model
    
    def get_background_means(self, img):
        """
        Inputs an image of the empty chessboard and returns the means of the V channel for each empty square.
        """
        empty_sqaures = proj.process_image_multiple_squares(img)

        gmm_empty_means = []

        for i, empty_square in enumerate(empty_sqaures):
            empty_square_hsv = cv2.cvtColor(empty_square, cv2.COLOR_BGR2HSV)
            value_channel = empty_square_hsv[:, :, 2].flatten()

            gmm = GaussianMixture(n_components=1)
            gmm.fit(value_channel.reshape(-1, 1))
            gmm_empty_means.append(gmm.means_.flatten()[0])

        return gmm_empty_means

    def predict_piece_color(self, square):
        """
        Predicts the color of the piece on the square.
        """

        init_square_hsv = cv2.cvtColor(square, cv2.COLOR_BGR2HSV)

        value_channel = init_square_hsv[:, :, 2].flatten()

        gmm_piece = GaussianMixture(n_components=2)
        gmm_piece.fit(value_channel.reshape(-1, 1))

        means_piece = gmm_piece.means_.flatten()
        mean_empty = self.square_background_means[i]

        piece_mean = max(means_piece, key=lambda x: abs(x - mean_empty))

        prediction = self.model_piece_color.predict([piece_mean])
        
        if prediction == 0:
            prediction = -1
        return prediction


    def get_piece_color_model(self, init_squares):
        """
        Inputs a list of 64 squares and returns a logistic regression model that determins if a square has a white or black piece."""
        piece_means = []

        for i, init_square in enumerate(init_squares):
            if i // 8 in [2, 3, 4, 5]:
                continue
            init_square_hsv = cv2.cvtColor(init_square, cv2.COLOR_BGR2HSV)

            value_channel = init_square_hsv[:, :, 2].flatten()

            gmm_piece = GaussianMixture(n_components=2)
            gmm_piece.fit(value_channel.reshape(-1, 1))

            means_piece = gmm_piece.means_.flatten()
            mean_empty = self.square_background_means[i]

            piece_mean = max(means_piece, key=lambda x: abs(x - mean_empty))
            piece_means.append(piece_mean)

        labels = get_init_piece_labels()
        model = get_logistic_regression_model(piece_means, labels)

        return model
    
    def fit_gmm(self, v_channel):
        """
        Fits a Gaussian Mixture Model to the V channel
        """
        gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
        gmm.fit(v_channel.reshape(-1, 1))

        return gmm

    def indentify_piece_ids(self, img):
        """
        Input the image from the camera and return -1 for each black piece, 0 for empty and 1 for white piece.
        """
        squares = proj.process_image_multiple_squares(img)

        # preprocess the squares (apply Gaussian blur and convert to HSV)
        blurred_hsv_squares = [preprocess_square(square) for square in squares]

        piece_v_empty = self.model_piece_vs_empty.predict(extract_features(blurred_hsv_squares))

        if self.piece_detection_algo == "standard":
            piece_squares = [square for i, square in enumerate(blurred_hsv_squares) if piece_v_empty[i] != 0]
            piece_label_ids = self.predict_piece_standard(piece_squares)
        elif self.piece_detection_algo == "red_green":
            piece_squares = [square for i, square in enumerate(blurred_hsv_squares) if piece_v_empty[i] != 0]
            piece_label_ids = self.predict_piece_red_green(piece_squares)
        elif self.piece_detection_algo == "GMM":
            piece_squares = [square for i, square in enumerate(squares) if piece_v_empty[i] != 0]
            piece_label_ids = self.predict_piece_GMM(piece_squares)

        board_ids = []
        for i in range(64):
            if piece_v_empty[i] == 0:
                board_ids.append(0)
            else:
                board_ids.append(piece_label_ids.pop(0))

        return board_ids   

    def predict_piece_standard(self, piece_squares):
        pixels = []
        for square in piece_squares:
            pixels.append(self.segment_piece(square))

        # Flatten the list of pixels
        all_pixels = np.hstack(pixels)

        # Fit a Gaussian Mixture Model to the V channel
        gmm = self.fit_gmm(all_pixels)

        white = gmm.means_.argmax()
        # classify the piece as black or white based on whether most of its pixels are closer to black or white
        piece_label_ids = [1 if gmm.predict(pixel.reshape(-1, 1)).sum() > pixel.size / 2 else 0 for pixel in pixels]

        piece_label_ids = [1 if id == white else -1 for id in piece_label_ids]
        

    def predict_piece_red_green(self, squares):
        """
        Create a classifier to determine if a square has a red or green piece.
        
        Parameters:
        squares: array[n] of 3d numpy arrays. Each square contains a chess piece.

        Returns:
        array[n] of ids. 1 is white, 0 is empty and -1 is black.
        """

        # Define HSV ranges for red and green colors
        lower_red_1 = np.array([0, 120, 90])
        upper_red_1 = np.array([35, 255, 255])
        lower_red_2 = np.array([165, 120, 90])
        upper_red_2 = np.array([180, 255, 255])
        lower_green = np.array([40, 90, 50])
        upper_green = np.array([75, 255, 255])

        piece_label_ids = []

        for i, square in enumerate(squares):
            # Count red and green pixels
            square_bgr = cv2.cvtColor(square, cv2.COLOR_HSV2BGR)
            debug.saveTempImg(square_bgr, f"square_{i}.jpg")
            red_mask = cv2.inRange(square, lower_red_1, upper_red_1)
            red_mask += cv2.inRange(square, lower_red_2, upper_red_2)
            green_mask = cv2.inRange(square, lower_green, upper_green)

            red_count = np.sum(red_mask)
            green_count = np.sum(green_mask)

            # Determine piece color based on the count
            if red_count > green_count:
                piece_label_ids.append(-1)  # Red piece (black ID)
            else:
                piece_label_ids.append(1)   # Green piece (white ID)

        return piece_label_ids
    
    def predict_piece_GMM(self, squares):

        piece_label_ids = []
        for square in squares:
            color = self.predict_piece_color()
            piece_label_ids.append(color)

        return piece_label_ids
        
    
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

    def print_masked_image(self, img, mask, labeled_pixels, segment_count):
        """
        Display the image with the mask applied.
        """
        flat_image = np.float32(img.reshape((-1,3)))
        # get the average color of each segment
        total = np.zeros((segment_count, 3), dtype=float)
        count = np.zeros(total.shape, dtype=float)
        for i, label in enumerate(labeled_pixels):
            if label == -1:
                continue
            total[label] = total[label] + flat_image[i]
            count[label] += 1
            
        avg = total/count
        avg = np.uint8(avg)

        # cast the labeled image into the corresponding average color
        green = np.array([125, 100, 50], dtype=np.uint8)
        res = np.array([avg[label] for label in labeled_pixels])
        res[mask] = green
        result = res.reshape((img.shape))
        result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)

        # proj.showImg(img)
        # proj.showImg(result)

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

def get_init_piece_labels():
    """
    Return an array [32] where 1 is a white piece, -1 is a black piece and 0 is empty.
    """
    labels = [1]*16+[0]*16

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