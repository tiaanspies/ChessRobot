from IK_Solvers.traditional import MotionPlanner
from Positioning.motor_commands import MotorCommandsSerial
import numpy as np
import matplotlib.pyplot as plt
from Data_analytics import correction_transform
from time import sleep
from pathlib import Path
from Camera import Camera_Manager
from Chessboard_detection import Aruco
import path_directories as dirs
import logging
import sys, time, datetime
import yaml

try:
    import plotly.graph_objects as go
except ModuleNotFoundError:
    print("analyze_transform: Did not load plotly, will not plot")

cm = MotionPlanner()
mc = MotorCommandsSerial()

def user_menu():
    """
    Create a user menu with the following options:
    1. Run calibration
    2. generate ideal calibration pattern
    3. generate transformed calibration pattern
    """

    print("\n1. Run calibration")
    print("2. Generate ideal calibration pattern")
    print("3. Generate transformed calibration pattern")
    print('4. Calculate Transformation Matrix')
    print("5. Find Home Position")
    print("0. Exit")

    choice = input("Select an option: ")

    if choice == "1":
        print("\nRunning calibration\n")
        run_calibration()
    elif choice == "2":
        print("\nGenerating ideal calibration pattern\n")
        generate_ideal_pattern()
    elif choice == "3":
        print("\nGenerating transformed calibration pattern\n")
        generate_transformed_pattern()
    elif choice == '4':
        print('Calculate Transformation Matrix')
        calculate_H_matrix()
    elif choice == "5":
        print("\nFinding Home Position\n")
        find_home_position()
    elif choice == "0":
        print("Exiting")
        exit()
    else:
        print("Invalid option")

def find_home_position():
    """Use camera to find calibrated home position"""

    aruco_tracker = create_tracker()

    # create camera object
    cam = Camera_Manager.RPiCamera(loadSavedFirst=False, storeImgHist=False)

    # Go to initial home position
    home_pos = cm.HOME.reshape((3,1)).astype(float)
    thetas = cm.inverse_kinematics(home_pos, True)
    print(f"Thetas: \n{thetas}, Home: \n{home_pos}")
    input("continue?")
    mc.filter_go_to(thetas, np.array([mc.GRIPPER_OPEN]))

    time.sleep(2)

    # get position in RCS
    ccs_current_pos = aruco_tracker.take_photo_and_estimate_pose(cam)
    ccs_control_pt_pos = cm.camera_to_control_pt_pos(ccs_current_pos)
    rcs_control_pt_pos = cm.ccs_to_rcs(ccs_control_pt_pos)
    
    error = cm.HOME.reshape((3,1)) - rcs_control_pt_pos
    print(f"RCS Control Point: \n{rcs_control_pt_pos}")
    error_norm = np.linalg.norm(error)

    print(f"Error: \n{error}; Error norm: {error_norm}")
    while error_norm > 5:
        # get position in RCS
        print(f"Thetas: {thetas}, Home: {home_pos}, Error: {error}")
        input("continue?")
        home_pos = home_pos + error * 0.6

        # move to new home position
        thetas = cm.inverse_kinematics(home_pos, True)
        mc.filter_go_to(thetas, np.array([mc.GRIPPER_OPEN]))

        time.sleep(2)

        # measure new position
        ccs_current_pos = aruco_tracker.take_photo_and_estimate_pose(cam)
        ccs_control_pt_pos = cm.camera_to_control_pt_pos(ccs_current_pos)
        rcs_control_pt_pos = cm.ccs_to_rcs(ccs_control_pt_pos)

        error = cm.HOME.reshape((3,1)) - rcs_control_pt_pos
        error_norm = np.linalg.norm(error)

        print(f"Error: {error}; Error norm: {error_norm}")

    print("Home position found")
    print(f"Thetas: {thetas}")

def fake_inverse_kinematics(path):
    return np.vstack((path,np.zeros_like(path[0,:])))

def draw_flat_cube(z, x_neg, x_pos, y_neg, y_pos):
    """Draws a flat cube"""
    step = 10
    path = np.hstack([
        cm.quintic_line(cm.HOME, np.array([x_neg, y_neg, z]), step),
        cm.quintic_line(np.array([x_neg, y_neg, z]), np.array([x_pos, y_neg, z]), step),
        cm.quintic_line(np.array([x_pos, y_neg, z]), np.array([x_pos, y_pos, z]), step),
        cm.quintic_line(np.array([x_pos, y_pos, z]), np.array([x_neg, y_pos, z]), step),
        cm.quintic_line(np.array([x_neg, y_pos, z]), np.array([x_neg, y_neg, z]), step),
        cm.quintic_line(np.array([x_neg, y_neg, z]), cm.HOME, step)
    ])

    return path

def draw_cube(v, slice_num):
    logging.info("Generating ideal pattern: Cube")
    logging.info(f"Vertices: {slice_num}")
    top_left = np.array([v["left"],v["close"],v["top"]])
    bottom_left = np.array([v["left"],v["close"],v["bottom"]])
    top_right = np.array([v["right"],v["close"],v["top"]])
    bottom_right = np.array([v["right"],v["close"],v["bottom"]])
    step = 10
    path = cm.quintic_line(cm.HOME, top_left, step)
    
    slice_width = (v["far"] - v["close"]) / slice_num
    slice_step = np.array([0.0, slice_width, 0.0], dtype=int)

    for i in range(slice_num):
        path = np.hstack((path, 
                          cm.quintic_line(top_left, bottom_right, step), 
                          cm.quintic_line(bottom_right, bottom_left, step), 
                          cm.quintic_line(bottom_left, top_right, step), 
                          cm.quintic_line(top_right, top_left, step)))
        if i < (slice_num)-1: # - 1):
            path = np.hstack((path, cm.quintic_line(top_left, top_left + slice_step, step)))
            top_left += slice_step
            top_right += slice_step
            bottom_left += slice_step
            bottom_right += slice_step
        
    path = np.hstack((path, cm.quintic_line(top_left, cm.HOME, step)))  

    return path

def create_tracker():
    # create aruco tracker object
    # use aruco_tracker to change default parameters
    aruco_obj = Aruco.ArucoTracker()

    # generate new pattern and save
    # aruco_obj.load_marker_pattern_positions(12, 17, 35, 26)
    aruco_obj.load_marker_pattern_positions(22, 30, 20, 15)

    return aruco_obj

def user_file_select(search_path:Path, message:str="Select a file: ", identifier:str="*_path_*"):
    """
    asks the user to select a file from any path that contains '*_path_*'.
    returns the prefix and suffix of the file name.
    """
    file_name_generator = search_path.glob(f"{identifier}")
    file_name_list = sorted([file_name for file_name in file_name_generator], reverse=True)

    # print the list of files
    print("\n"+message)
    for i, file_name in enumerate(file_name_list):
        print(f"{i}: {file_name.name}")

    # get user input
    user_input = input("Enter a number: ")
    user_input = int(user_input)

    # return the selected file name
    name = file_name_list[user_input].stem
    prefix = name.split(identifier.strip("*"))[0]  # remove everything after and including "_path_"
    suffix = name.split(identifier.strip("*"))[-1]  # remove everything before and including "_path_"

    return prefix, suffix

def user_file_select_multiple(search_path:Path, message:str="Select a file: ", identifier:str="*_path_*"):
    """
    asks the user to select a file from any path that contains '*_path_*'.
    returns the prefix and suffix of the file name.
    """
    file_name_generator = search_path.glob(f"{identifier}")
    file_name_list = sorted([file_name for file_name in file_name_generator], reverse=True)

    # print the list of files
    print(message)
    for i, file_name in enumerate(file_name_list):
        print(f"{i}: {file_name.name}")

    # get user input
    user_input = input("Enter a number or q to continue: ")
    selected_file_prefixes = []
    selected_file_suffixes = []

    while user_input != "q":
        
        user_input = int(user_input)

        # return the selected file name
        name = file_name_list[user_input].stem
        selected_file_prefixes.append(name.split(identifier.strip("*"))[0])  # remove everything after and including "_path_"
        selected_file_suffixes.append(name.split(identifier.strip("*"))[-1])  # remove everything before and including "_path_"

        user_input = input("Enter a number or q to continue: ")

    return selected_file_prefixes, selected_file_suffixes


def contains_nan(arr):
    return np.isnan(arr).any()


def run_and_track(tracker: Aruco.ArucoTracker, cam, cal_path: Path):
    """
    Main function for moving to all the calibration points and tracking them.
    """
    # load path
    dimensions, transform_type = user_file_select(dirs.PLANNED_PATHS)

    # look for all files that match the selected file.
    plan_cartesian = [f for f in dirs.PLANNED_PATHS.glob(f"{dimensions}_path_{transform_type}.npy")]
    plan_ja = [f for f in dirs.PLANNED_PATHS.glob(f"{dimensions}_ja_{transform_type}.npy")]

    assert len(plan_cartesian) == 1 and len(plan_ja) == 1, "Multiple or no matching files found. \n Have Tiaan fix his code"

    # Should only be one file, load it
    plan_cartesian = np.load(plan_cartesian[0])
    plan_ja = np.load(plan_ja[0])
   
    # Load path into motion controller
    mc.load_path(plan_ja, plan_cartesian)

    # init counters
    moves_total = plan_ja.shape[1]
    moves_current = 0

    # Initialize tracking variables
    measured_cartesian = np.full(plan_cartesian.shape, np.nan)

    # move to init position

    mc.go_to(plan_ja[:, 0])

    # step through
    run_cal = True 
    while run_cal:

        # Move to next position
        run_cal, _ = mc.run_once(move_time=400)
        sleep(1)
        
        # attempt twice to take photo
        iter = 0
        ccs_current_pos = tracker.take_photo_and_estimate_pose(cam)
        while iter < 2 and contains_nan(ccs_current_pos):
            logging.debug("Failed to take photo, trying again.")
            sleep(1)
            ccs_current_pos = tracker.take_photo_and_estimate_pose(cam)
            sleep(1)
            iter += 1

        ccs_control_pt_pos = cm.camera_to_control_pt_pos(ccs_current_pos)
        rcs_control_pt_pos = cm.ccs_to_rcs(ccs_control_pt_pos)
        
        measured_cartesian[:, [moves_current]] = rcs_control_pt_pos
            
        logging.debug(f"Position: {ccs_current_pos.reshape(1,3)}")
        sleep(0.2)

        moves_current += 1
        print(f"Progress: {moves_current/moves_total*100:.2f}%")

    # save data
    prefix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    logging.info("Saving recorded data.")
    np.save(Path(cal_path, prefix + "_measured.npy"), measured_cartesian)
    np.save(Path(cal_path, prefix + "_planned_path.npy"), plan_cartesian)

def run_calibration():
    """
    Run the calibration process
    """
    # load aruco obj things
    aruco_obj = create_tracker()

    # create camera object
    cam = Camera_Manager.RPiCamera(loadSavedFirst=False, storeImgHist=False)

    # calibration path
    run_and_track(aruco_obj, cam, dirs.CAL_TRACKING_DATA_PATH)

def generate_ideal_pattern():
    """
    Generate an ideal calibration pattern
    """
    vertices = {
    "top" : 260,
    "bottom" : 180,
    "right" : 160,
    "left" : -160,
    "close" : 150,
    "far" : 420}

    print("Did you update the code to make the new shape? (Y/N)")
    if input() != "Y":
        print("Please update the code to make the new shape.")
        return

    # generate waypoints
    path = draw_cube(vertices, 4) 
    # path = draw_flat_cube(125, -90, 90, 150, 350)

    # # create name
    name = "_".join([str(v) for v in vertices.values()])
    # name = "flat_cube_125_-90_90_150_250"
    name_path = name+"_path_ideal"
    name_joint_angles = name+"_ja_ideal"

    logging.info("solving for joint angles (Inverse Kinematics).")
    start_time = time.time()
    thetas = cm.inverse_kinematics(path, apply_compensation=False) # convert to joint angles
    end_time = time.time()

    logging.info(f"Time taken: {end_time - start_time:.2f} seconds")

    logging.info("Adding gripper commands")
    grip_commands = cm.get_gripper_commands2(path) # remove unnecessary wrist commands, add gripper open close instead
    joint_angles, exceeds_lim = mc.sort_commands(thetas, grip_commands)

    logging.info("Correcting joint angles")
    joint_angles, path = mc.correct_limits(joint_angles, path, exceeds_lim)
    
    logging.info(f"Saving path as '{name_path}'")
    np.save(Path(dirs.PLANNED_PATHS, name_path), path)
    
    logging.info(f"Saving joing angles as '{name_joint_angles}'")
    np.save(Path(dirs.PLANNED_PATHS, name_joint_angles), joint_angles)

def H_matrix_validity_range(pts_ideal):
    """Finds the min and max values for each axis of points used in H matrix calculation"""

    #TODO: Make work with multiple H matrices, currently it will return incorrect result with cascaded H matrices
    min_x, max_x = np.min(pts_ideal[0]), np.max(pts_ideal[0])
    min_y, max_y = np.min(pts_ideal[1]), np.max(pts_ideal[1])
    min_z, max_z = np.min(pts_ideal[2]), np.max(pts_ideal[2])

    range_dict = {"min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y, "min_z": min_z, "max_z": max_z}
    return range_dict

def save_H_matrix_validity_range(range_dict, file_path):
    """Saves the transformation matrix validity range as a yaml file."""

    with open(file_path, 'w') as file:
        yaml.dump(range_dict, file)

def calculate_H_matrix():
    """Calculates H between measured and ideal points"""
    
    message = "Select ideal path"
    file_prefix, suffix = user_file_select(dirs.PLANNED_PATHS, message, '*_path_*')
    
    #Load ideal points
    name_ideal = file_prefix+"_path_"+suffix+'.npy'
    file_ideal = Path(dirs.PLANNED_PATHS, name_ideal)
    pts_ideal = np.load(file_ideal)

    # Ask user which file to use for real points
    message = "Select file for Transformation matrix"
    file_prefix, suffix = user_file_select(dirs.CAL_TRACKING_DATA_PATH, message, '*_measured*')
    
    # load real and planned points
    name_real = file_prefix+"_measured.npy"
    file_real = Path(dirs.CAL_TRACKING_DATA_PATH, name_real)
    pts_real = np.load(file_real)
    
    # remove points that contain NAN
    mask = ~(np.isnan(pts_real).any(axis=0) | np.isnan(pts_ideal).any(axis=0))
    pts_ideal = pts_ideal[:, mask]
    pts_real = pts_real[:, mask]
    
    # Calculate transformation matrix
    H = correction_transform.attempt_minimize_quad(pts_ideal, pts_real)
    print(f"Saving as {file_prefix}_H_matrix{suffix}")

    # Save transformation matrix
    H_path = Path(dirs.H_MATRIX_PATH, file_prefix+"_H_matrix"+suffix+'.csv')
    correction_transform.save_transformation_matrix(H_path, H)
    
    # Save the validity range of the transformation matrix
    range_path = Path(dirs.H_MATRIX_PATH, file_prefix+"_H_matrix"+suffix+'_valid_range.yaml')
    range_dict = H_matrix_validity_range(pts_ideal)
    save_H_matrix_validity_range(range_dict, range_path)
    
def generate_transformed_pattern():
    """
    Generate a transformed calibration pattern
    """
    # Get transformation matrix
    message = "Select transformation Matrix"
    file_prefixes, suffixes = user_file_select_multiple(dirs.H_MATRIX_PATH, message, '*_H_matrix*')
    paths = [Path(dirs.H_MATRIX_PATH, f"{p}_H_matrix{s}.csv") for p, s in zip(file_prefixes, suffixes)]
    H_list = correction_transform.load_transformation_matrix_multiple(paths)

    # Get ideal path
    message="\nWhich base path would you like to transform?"
    ideal_prefix, ideal_suffix = user_file_select(dirs.PLANNED_PATHS, message,"*_path_ideal*")
    pts_ideal = np.load(Path(dirs.PLANNED_PATHS, f"{ideal_prefix}_path_ideal{ideal_suffix}.npy"))

    # get the current time for a prefix
    prefix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    name_joint_angles_transformed = prefix+"_ja_transformed"+ideal_suffix
    name_path_transformed = prefix+"_path_transformed"+ideal_suffix
    
    compensated_points = correction_transform.project_points_quad_multiple(pts_ideal, H_list)

    # solve inverse kinematics
    logging.info("solving inverse kinematics...")
    thetas = cm.inverse_kinematics(compensated_points, True) # convert waypoints to joint angles
    joint_angles, exceeds_lim = mc.sort_commands(thetas, None)

    logging.info("Correcting joint angles")
    joint_angles, compensated_points = mc.correct_limits(joint_angles, compensated_points, exceeds_lim)
    logging.info("Solved inverse kinetatics. Saving...")

    # if platform.system() == "Windows":
    #     cm.plot_robot(thetas, compensated_points)
    print(f"SHAPE: {compensated_points.shape}")
    print(f"SHAPE JOINTS: {joint_angles.shape}")
    np.save(Path(dirs.PLANNED_PATHS, name_path_transformed), compensated_points)
    np.save(Path(dirs.PLANNED_PATHS, name_joint_angles_transformed), joint_angles)

if __name__ == "__main__":
    log_level = sys.argv[1] if len(sys.argv) > 1 else "Debug"

    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % log_level)
    logging.basicConfig(level=logging.DEBUG)

    user_menu()
