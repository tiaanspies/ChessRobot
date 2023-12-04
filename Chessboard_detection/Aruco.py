"""
My aruco file
"""

import cv2
import cv2.aruco as aruco
import numpy as np

try:
    from Chessboard_detection import pi_debugging
    from Chessboard_detection import Camera_Manager
except ModuleNotFoundError:
    import pi_debugging
    import Camera_Manager

import json
from pathlib import Path

MAJOR = 4
MINOR = 6

class ArucoTracker:
    """
    A tracking object for the custom aruco boards
    """
    def __init__(self, aruco_position_path_dir: str, aruco_dict) -> None:
        self.marker_path_dir = aruco_position_path_dir
        self.marker_positions = None
        self.aruco_dict = aruco_dict

        if Path(self.marker_path_dir).is_dir() == False:
            raise ValueError("Marker path directory does not exist")

    def generate_calibio_pattern_positions(self, rows:int, cols:int, checker_size:float, marker_size:float):   
        """
        Creates a list of marker positions for the calibration pattern
        """   
        edge_size = (checker_size-marker_size)/2
        
        # create a list with 4x2 list for each marker
        # each marker will store the x and y position for its 4 corners
        positions = [[[0.0,0.0], [0.0,0.0], [0.0,0.0], [0.0,0.0]] for i in range(int(rows*cols/2))]
        
        # create positions for each marker
        # markers are placed in a zigzag pattern
        for x in range(int(rows)):
            # odd rows might have one less marker
            if x % 2 == 0:
                y_target = int(cols/2)+1
            else:
                y_target = int(cols/2)

            for y in range(y_target):
                if x % 2 == 0:
                    marker_pos = [edge_size+x*checker_size, edge_size+y*checker_size*2]
                else:
                    marker_pos = [edge_size+x*checker_size, edge_size+(y+0.5)*checker_size*2]

                # create positions for each corner
                positions[calc_square_id(x, y, cols)][0] = marker_pos
                positions[calc_square_id(x, y, cols)][1] = [marker_pos[0], marker_pos[1]+marker_size]
                positions[calc_square_id(x, y, cols)][2] = [marker_pos[0]+marker_size, marker_pos[1]+marker_size]
                positions[calc_square_id(x, y, cols)][3] = [marker_pos[0]+marker_size, marker_pos[1]]

        # save position data
        properties = [rows, cols, checker_size, marker_size]
        aruco_data = [properties, positions]
        
        return aruco_data

    def load_marker_pattern_positions(self, rows:int, cols:int, checker_size:float, marker_size:float) -> None:
        """
        Checks whether the marker pattern exists and creates it if it does not.
        """
        properties = [rows, cols, checker_size, marker_size]
        properties_name = "_".join([str(i) for i in properties])
        properties_path = Path(self.marker_path_dir, properties_name+".json")

        # check if marker pattern exists
        if properties_path.is_file() == False:
            # generate marker pattern
            aruco_data = self.generate_calibio_pattern_positions(rows, cols, checker_size, marker_size)
            
            properties = aruco_data[0]
            self.marker_positions = aruco_data[1]

            # write the aruco data to a json file
            with open(properties_path, "w") as json_file:
                json.dump(aruco_data, json_file, indent=4)

            print(f"Wrote marker properties to {self.marker_path_dir}\\{properties_name}.json")
        else:
            # load marker pattern
            with open(properties_path, "r") as json_file:
                data = json.load(json_file)
                self.marker_positions = data[1]

            if properties != data[0]:
                raise ValueError("Marker properties do not match")

        # convert to numpy array
        for pos in range(self.marker_positions.__len__()):
            self.marker_positions[pos] = np.array(self.marker_positions[pos])

    def take_photo_and_estimate_pose(self, cam: Camera_Manager.RPiCamera)-> tuple[np.ndarray, np.ndarray]:
        # Camera parameters
        camera_matrix, dist_coeffs = cam.readCalibMatrix()

        # Load the image
        _, image = cam.read()

        ids, corners = detect_markers(self.aruco_dict, image)

        if ids is None:
            print("No markers found")
            return None, None
        
        rvecs, tvecs = estimate_pose(corners, ids, camera_matrix, dist_coeffs, self.marker_positions)
        
        if rvecs is not None:
            # rvecs and tvecs contain the rotation and translation vectors for each detected marker
            print("Rotation vectors:")
            print(rvecs)

            print("Translation vectors:")
            print(tvecs)  
            
            # Draw the detected markers and axes on the image
            # for id in ids:
            #     cv2.aruco.drawDetectedMarkers(image, corners)
                # cv2.aruco.drawAxis(image, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], marker_size * 0.5)

            # Draw the origin on the board
            cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvecs, tvecs, 100, 3)
            label_markers(image, ids, corners)
            pi_debugging.saveTempImg(image, "Aruco_markers_w_axes.png")

        return rvecs, tvecs


def main():
    # define aruco pattern file location.
    # name = "7x5_small_3x2_large_4x4_50_marker"
    dir_path = Path("Chessboard_detection", "Aruco Markers").resolve().__str__()

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)

    # # create camera object
    cam = Camera_Manager.RPiCamera(loadSavedFirst=False)

    # # create aruco tracker object
    aruco_obj = ArucoTracker(dir_path, aruco_dict)

    # aruco_obj.load_marker_pattern_positions(14, 20, 30, 22)
    aruco_obj.load_marker_pattern_positions(12, 17, 35, 26)

    # get camera position
    aruco_obj.take_photo_and_estimate_pose(cam)
    
    # track(file_path_and_name, size_correction)
    
    # ids, corners = detect_markers(aruco_dict)

    # label_markers(image, ids, corners)

def calc_square_id(row:int, col:int, cols:int):
    # return the id of the square
    cols_d2 = int(cols/2)
    if cols % 2 == 0:
        return row*cols_d2 + col
    else:
        return int((row+1)/2)*(cols_d2+1) + int(row/2)*cols_d2 + col

def generate_pattern(page: np.ndarray, rows:int, cols:int, dict, count_start: int):
    page_height, page_width = page.shape
    
    # percentage of marker col to fill with marker
    marker_size_ratio = 0.8

    # create marker size based on width of page dimension
    marker_size_width = int(marker_size_ratio * page_width / (2*cols+1))
    marker_size_height = int(marker_size_ratio * page_height / (2*rows+1))

    marker_size = min(marker_size_width, marker_size_height)
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
            if count >= 250:
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

            positions[count] = pos.tolist()
        
    total_length = int(row_space+marker_size/2+(rows-1)*(marker_size+row_space)+w) - int(row_space)

    return count, positions, total_length

def label_markers(image, ids:list, corners:list) -> np.ndarray:
    """
    Draws and labels the detected markers on the image
    """
    if ids is not None:
        for i in range(len(ids)):
            # Draw the bounding box around the marker
            aruco.drawDetectedMarkers(image, corners)

            # Draw the marker ID
            c = corners[i][0]
            x, y = int(c[:, 0].mean()), int(c[:, 1].mean())
            cv2.putText(image, str(ids[i][0]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    return image

def detect_markers(aruco_dict: cv2.aruco_Dictionary, image: np.ndarray)-> tuple[list, list]:
    """
    Find aruco markers and return their ids and corner positions
    """

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

    return ids, corners

def estimate_pose(corners, ids, camera_matrix, dist_coeffs, marker_positions
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    if ids is None:
        return None, None, None

    # create world and image point matrices
    world_points = np.zeros((0,2))
    image_points = np.zeros((0,2))
    for id, corner_set in zip(ids.flatten(), corners):
        if id >= 102:
            print("fix this line")
            continue

        image_points = np.vstack([image_points, corner_set[0]])
        world_points = np.vstack([world_points, marker_positions[id]])

    # add colum of zeros to world_points
    world_points = np.hstack([world_points, np.zeros((world_points.shape[0], 1))])

    #check that at least 1 marker is found
    if ids is None:
        print("No Aruco markers found.")
        return None, None, None
    
    # Estimate pose for each detected marker
    res, rvecs, tvecs= cv2.solvePnP(world_points, image_points, camera_matrix, dist_coeffs)

    # check that localization succeeded
    if res == False:
        raise ValueError("solvePnP failed to localize using detected Aruco Markers")

    return rvecs, tvecs

if __name__ == '__main__':
    main()