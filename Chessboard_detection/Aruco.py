"""
My aruco file
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import pi_debugging
import Camera_Manager
from pathlib import Path

MAJOR = 4
MINOR = 6

PIXEL_TO_MM = 25.4*8.5/816

def main():
    name = "7x5_small_3x2_large"
    file_path_and_name = Path("Chessboard_detection", "Aruco Markers", name).resolve().__str__()

    generate(file_path_and_name)
    size_correction = 186.5/191.29374999999996
    track(file_path_and_name, size_correction)

def generate_pattern(page: np.ndarray, rows:int, cols:int, dict, count_start: int):
    page_height, page_width = page.shape
    
    # percentage of marker col to fill with marker
    marker_size_ratio = 0.8

    # create marker size based on width of page dimension
    marker_size = int(marker_size_ratio * page_width / (2*cols+1))

    if marker_size % 2 == 1:
        marker_size -= 1

    # count for each marker ID
    # positions will store the edge positions of the markers
    count = count_start
    positions = {}

    w = int(marker_size/2)
    for row in range(rows):
        for col in range(cols):
            count += 1
            if count >= 50:
                break
            
            # create array for one marker
            marker_image = np.zeros((marker_size, marker_size), dtype=np.uint8)

            if MAJOR > 4 or (MAJOR == 4 and MINOR >= 7):
                marker_image = aruco.generateImageMarker(dict, count, marker_size, 20)
            else:
                marker_image = aruco.drawMarker(dict, count, marker_size, 20)

            # calculate whitespace between markers
            row_space = (page_height-rows*marker_size)/(rows+1)
            col_space = (page_width-cols*marker_size)/(cols+1)

            # Find center points for the marker
            col_center = int(col_space+marker_size/2+col*(marker_size+col_space))
            row_center = int(row_space+marker_size/2+row*(marker_size+row_space))

            # check if space is free to create marker
            if (page[row_center-w:row_center+w, col_center-w:col_center+w]==255).all():
                page[row_center-w:row_center+w, col_center-w:col_center+w] = marker_image

            # Store the positions of the marker corners
            pos = np.array([[col_center - w, -row_center + w, 0]])
            pos = np.vstack([pos, [col_center + w, -row_center + w, 0]])
            pos = np.vstack([pos, [col_center + w, -row_center - w, 0]])
            pos = np.vstack([pos, [col_center - w, -row_center - w, 0]])

            positions[count] = pos
        
    total_length = int(row_space+marker_size/2+(rows-1)*(marker_size+row_space)+w) - int(row_space)

    return count, positions, total_length

def generate(dest_path):    
    # Define the size and number of bits of the marker
    page = np.ones((1056, 816))*255
    count = 0
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

    # generate large markers
    count, positions_large, total_length = generate_pattern(page,3, 2, dictionary, count)
    print(f"Predicted large marker width (mm): {total_length*PIXEL_TO_MM}")
    # generate small markers
    count, positions_small, total_length = generate_pattern(page, 7, 5, dictionary, count)
    print(f"Predicted small marker width (mm): {total_length*PIXEL_TO_MM}")

    # join position dictionaries
    positions = positions_small | positions_large

    # Display the generated marker
    # cv2.imshow("Marker Image", page)
    # cv2.waitKey(0)
    # # Save the marker image to a file
    cv2.imwrite(dest_path+".png", page)

    # save position data
    np.save(dest_path+".npy", positions)

def detect_and_estimate_pose(image, camera_matrix, dist_coeffs, marker_positions):
    # Load the ArUco dictionary
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

   # new version of OPENCV
    if MAJOR > 4 or (MAJOR == 4 and MINOR >= 7):
        # Create the ArUco parameters
        aruco_params = cv2.aruco.DetectorParameters()

        #create detector
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

        # Detect markers in the image
        corners, ids, _ = detector.detectMarkers(image)
    else: #OLD version of opencv
        # create detector
        aruco_params = cv2.aruco.DetectorParameters_create()

        # detect markers
        corners, ids, _ = cv2.aruco.detectMarkers(image, aruco_dict, parameters=aruco_params)

    # create world and image point matrices
    world_points = np.zeros((0,3))
    image_points = np.zeros((0,2))
    for id, corner_set in zip(ids.flatten(), corners):
        image_points = np.vstack([image_points, corner_set[0]])
        world_points = np.vstack([world_points, marker_positions[id]])

    #check that at least 1 marker is found
    if ids is None:
        print("No Aruco markers found.")
        return None, None, None
    
    # Estimate pose for each detected marker
    res, rvecs, tvecs= cv2.solvePnP(world_points, image_points, camera_matrix, dist_coeffs)

    # check that localization succeeded
    if res == False:
        raise ValueError("solvePnP failed to localize using detected Aruco Markers")

    # Draw the detected markers and axes on the image
    for id in ids:
        cv2.aruco.drawDetectedMarkers(image, corners)
        # cv2.aruco.drawAxis(image, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], marker_size * 0.5)

    # Draw the origin on the board
    cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvecs, tvecs, 100, 3)
    return rvecs, tvecs, image

def track(file_name_and_path, size_correction):
    cam = Camera_Manager.RPiCamera(loadSavedFirst=False)

    # scale to convert from aruco size in pixels to mm
    scale_conv = PIXEL_TO_MM*size_correction

    # Camera parameters (replace these with your own calibration values)
    camera_matrix, dist_coeffs = cam.readCalibMatrix()

    #read position data
    marker_positions = np.load(file_name_and_path+".npy", allow_pickle=True).flat[0]
    for pos in marker_positions:
        marker_positions[pos] = marker_positions[pos]*scale_conv
    # *

    # Load the image
    _, image = cam.read()
    # image = cv2.imread("large.png")

    rvecs, tvecs, image_with_axes = detect_and_estimate_pose(image, camera_matrix, dist_coeffs, marker_positions)

    if rvecs is not None:
        # rvecs and tvecs contain the rotation and translation vectors for each detected marker
        print("Rotation vectors:")
        print(rvecs)

        print("Translation vectors:")
        print(tvecs)

        # Display the image with the detected markers and axes
        # cv2.imshow("Camera Pose Estimation", image_with_axes)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

if __name__ == '__main__':
    main()