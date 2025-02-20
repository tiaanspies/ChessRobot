import numpy as np
from time import sleep
from pathlib import Path
from Camera import Camera_Manager
from Chessboard_detection import Aruco
import path_directories as dirs
import sys, time, datetime, yaml, logging
import Positioning.robot_manager as robot_manager
import Data_analytics.correction_transform as correction_transform
from path_directories import CONFIG_PATH_KINEMATICS

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
        run_and_track(dirs.CAL_TRACKING_DATA_PATH)
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
        find_overview_positions()
    elif choice == "0":
        print("Exiting")
        exit()
    else:
        print("Invalid option")

def find_overview_positions():
    
    position_names = ["home_position", "home_pos_forward", "home_pos_backward"]

    for position_name in position_names:
        print(f"Finding {position_name}")
        find_target_position(position_name)

def find_target_position(position_name):
    """Use camera to find calibrated target position"""
    
    robot = robot_manager.Robot()

    # Load target position to calibrate
    config = yaml.safe_load(open(CONFIG_PATH_KINEMATICS))['IK_CONFIG']
    target_position = np.array([
        [config[position_name]['x']],
        [config[position_name]['y']],
        [config[position_name]['z']]
    ], dtype=np.float32)

    # target pos offset is the position to move back and forth to.
    # Moving away and back is meant to help overcome static friction.
    target_pos_compensated = target_position.copy()
    target_pos_offset = target_pos_compensated - np.array([[0], [100], [100]])

    # move to offset position before actual to overcome static friction.
    robot.move_to_single(target_pos_offset, robot.motor_commands.GRIPPER_OPEN, apply_compensation=False)
    robot.move_to_single(target_pos_compensated, robot.motor_commands.GRIPPER_OPEN, apply_compensation=False)

    # get position in RCS using camera and aruco board
    sleep(1)
    rcs_control_pt_pos = robot.get_rcs_pos_aruco()
    
    error = target_position - rcs_control_pt_pos
    error_norm = np.linalg.norm(error)

    print(f"Error: \n{error}; Error norm: {error_norm}")
    iter = 0
    while error_norm > 4 and iter < 20:
        # get position in RCS
        target_pos_compensated += error * 0.6
        iter += 1

        # move to new target position
        robot.move_to_single(target_pos_offset, robot.motor_commands.GRIPPER_OPEN, apply_compensation=False)
        robot.move_to_single(target_pos_compensated, robot.motor_commands.GRIPPER_OPEN, apply_compensation=False)

        # measure new position
        sleep(2)
        rcs_control_pt_pos = robot.get_rcs_pos_aruco()

        error = target_position - rcs_control_pt_pos
        error_norm = np.linalg.norm(error)

        print(f"Target_position: \n{target_pos_compensated}\n Position: \n{rcs_control_pt_pos}\nError: \n{error}\nError norm: \n{error_norm}")

    joint_angles = robot.motion_planner.inverse_kinematics(target_pos_compensated, apply_compensation=False)
    print("Target position found")
    print(f"ja: {joint_angles}")

    # Save target calibration
    config = yaml.safe_load(open(CONFIG_PATH_KINEMATICS, 'r'))
    ja = config["IK_CONFIG"][position_name+"_joint_angles"]
    ja["base"] = float(joint_angles[0,0])
    ja["shoulder"] = float(joint_angles[1,0])
    ja["elbow"] = float(joint_angles[2,0])
    config["IK_CONFIG"][position_name+"_joint_angles"] = ja

    with open(CONFIG_PATH_KINEMATICS, 'w') as file:
        yaml.dump(config, file)

def fake_inverse_kinematics(path):
    return np.vstack((path,np.zeros_like(path[0,:])))

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

def run_and_track(cal_path: Path):
    """
    Main function for moving to all the calibration points and tracking them.
    """
    robot = robot_manager.Robot()
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
    robot.motor_commands.load_path(plan_ja, plan_cartesian)

    # init counters
    moves_total = plan_ja.shape[1]
    moves_current = 0

    # Initialize tracking variables
    measured_cartesian = np.full(plan_cartesian.shape, np.nan)

    # move to init position

    robot.motor_commands.go_to(plan_ja[:, 0])

    # step through
    run_cal = True 
    while run_cal:

        # Move to next position
        run_cal, _ = robot.motor_commands.run_once(move_time=400)
        sleep(1)
        
        # attempt twice to take photo
        iter = 0
        rcs_control_pt_pos = robot.get_rcs_pos_aruco()
        while iter < 2 and contains_nan(rcs_control_pt_pos):
            logging.debug("Failed to take photo, trying again.")
            sleep(1)
            rcs_control_pt_pos = robot.get_rcs_pos_aruco()
            sleep(1)
            iter += 1
        
        measured_cartesian[:, [moves_current]] = rcs_control_pt_pos
            
        logging.debug(f"Position: {rcs_control_pt_pos.reshape(1,3)}")
        sleep(0.2)

        moves_current += 1
        print(f"Progress: {moves_current/moves_total*100:.2f}%")

    # save data
    prefix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    logging.info("Saving recorded data.")
    np.save(Path(cal_path, prefix + "_measured.npy"), measured_cartesian)
    np.save(Path(cal_path, prefix + "_planned_path.npy"), plan_cartesian)

def generate_ideal_pattern():
    """
    Generate an ideal calibration pattern
    """

    robot = robot_manager.Robot()
    vertices = {
        "top" : 260,
        "bottom" : 180,
        "right" : 160,
        "left" : -160,
        "close" : 150,
        "far" : 420
    }

    print("Did you update the code to make the new shape? (Y/N)")
    if input().upper() != "Y":
        print("Please update the code to make the new shape.")
        return

    # generate waypoints
    path = robot.motion_planner.draw_cube(vertices, 4)
    # path = draw_flat_cube(125, -90, 90, 150, 350)

    # # create name
    name = "_".join([str(v) for v in vertices.values()])
    # name = "flat_cube_125_-90_90_150_250"
    name_path = name+"_path_ideal"
    name_joint_angles = name+"_ja_ideal"

    logging.info("solving for joint angles (Inverse Kinematics).")
    start_time = time.time()
    thetas = robot.motion_planner.inverse_kinematics(path, apply_compensation=False) # convert to joint angles
    end_time = time.time()

    logging.info(f"Time taken: {end_time - start_time:.2f} seconds")

    logging.info("Adding gripper commands")
    joint_angles, exceeds_lim = robot.motor_commands.sort_commands(thetas, None)

    logging.info("Correcting joint angles")
    joint_angles, path = robot.motor_commands.correct_limits(joint_angles, path, exceeds_lim)
    
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
    robot = robot_manager.Robot()
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
    thetas = robot.motion_planner.inverse_kinematics(compensated_points, True) # convert waypoints to joint angles
    joint_angles, exceeds_lim = robot.motor_commands.sort_commands(thetas, None)

    logging.info("Correcting joint angles")
    joint_angles, compensated_points = robot.motor_commands.correct_limits(joint_angles, compensated_points, exceeds_lim)
    logging.info("Solved inverse kinetatics. Saving...")

    # if platform.system() == "Windows":
    #     cm.plot_robot(thetas, compensated_points)
    print(f"SHAPE: {compensated_points.shape}")
    print(f"SHAPE JOINTS: {joint_angles.shape}")
    np.save(Path(dirs.PLANNED_PATHS, name_path_transformed), compensated_points)
    np.save(Path(dirs.PLANNED_PATHS, name_joint_angles_transformed), joint_angles)

def run(log_level):
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % log_level)
    logging.basicConfig(level=logging.DEBUG)

    user_menu()

if __name__ == "__main__":
    log_level = sys.argv[1] if len(sys.argv) > 1 else "Debug"

    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % log_level)
    logging.basicConfig(level=logging.DEBUG)

    user_menu()
