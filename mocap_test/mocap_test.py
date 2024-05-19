from IK_Solvers.traditional import ChessMoves
from Positioning.motor_commands import MotorCommands
import numpy as np
import matplotlib.pyplot as plt
from Data_analytics import correction_transform
from time import sleep
from pathlib import Path
from Camera import Camera_Manager
from Chessboard_detection import Aruco
import datetime
import cv2.aruco as aruco
import path_directories as dirs
from Data_analytics import analyze_transform
import platform
import logging
import sys

try:
    import plotly.graph_objects as go
except ModuleNotFoundError:
    print("analyze_transform: Did not load plotly, will not plot")

cm = ChessMoves()
mc = MotorCommands()

def main():
    # load aruco obj things
    aruco_obj = create_tracker()

    # # create camera object
    cam = Camera_Manager.RPiCamera(loadSavedFirst=False, storeImgHist=False)

    run_and_track(aruco_obj, cam, dirs.CAL_TRACKING_DATA_PATH)

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
    logging.info(f"Vertices: {vertices}")
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

def run_and_track(tracker: Aruco.ArucoTracker, cam, cal_path: Path):
    # load path
    dimensions, transform_type = user_file_select(dirs.PLANNED_PATHS)

    plan_cartesian = [f for f in dirs.PLANNED_PATHS.glob(f"{dimensions}_path_{transform_type}.npy")]
    plan_ja = [f for f in dirs.PLANNED_PATHS.glob(f"{dimensions}_ja_{transform_type}.npy")]

    assert len(plan_cartesian) == 1 and len(plan_ja) == 1, "Multiple or no matching files found. \n Have Tiaan fix his code"

    plan_cartesian = np.load(plan_cartesian[0])
    plan_ja = np.load(plan_ja[0])
   
    # Load path into motion controller
    mc.load_path(plan_ja, plan_cartesian)

    # init counters
    moves_total = plan_ja.shape[1]
    moves_current = 0

    # Initialize tracking variables
    measured_cartesian = np.full(plan_cartesian.shape, np.nan)

    # step through
    run_cal = True 
    while run_cal:

        # Move to next position
        run_cal, plan_points = mc.run_once()
        sleep(2)
        
        # get position
        ccs_current_pos = tracker.take_photo_and_estimate_pose(cam)
        ccs_control_pt_pos = cm.camera_to_control_pt_pos(ccs_current_pos)
        rcs_control_pt_pos = cm.ccs_to_rcs(ccs_control_pt_pos)
        
        measured_cartesian[:, [moves_current]] = rcs_control_pt_pos
            
        logging.debug(f"Position: {ccs_current_pos.reshape(1,3)}")
        sleep(1)

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

    # # create camera object
    cam = Camera_Manager.RPiCamera(loadSavedFirst=False, storeImgHist=False)

    # calibration path
    run_and_track(aruco_obj, cam, dirs.CAL_TRACKING_DATA_PATH)

def generate_ideal_pattern():
    """
    Generate an ideal calibration pattern
    """
    vertices = {
    "top" : 200,
    "bottom" : 100,
    "right" : 160,
    "left" : -160,
    "close" : 110,
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
    thetas = cm.inverse_kinematics(path) # convert to joint angles

    logging.info("Adding gripper commands")
    grip_commands = cm.get_gripper_commands2(path) # remove unnecessary wrist commands, add gripper open close instead
    joint_angles, exceeds_lim = mc.sort_commands(thetas, grip_commands)

    logging.info("Correcting joint angles")
    joint_angles, path = mc.correct_limits(joint_angles, path, exceeds_lim)
    
    logging.info(f"Saving path as '{name_path}'")
    np.save(Path(dirs.PLANNED_PATHS, name_path), path)
    
    logging.info(f"Saving joing angles as '{name_joint_angles}'")
    np.save(Path(dirs.PLANNED_PATHS, name_joint_angles), joint_angles)

def go_to(pos):
    """Take an input position, find the joint angles and go to that position"""

    if pos.shape == np.array((3,)).shape:
        pos = pos.reshape(3,1)

    thetas = cm.inverse_kinematics(pos)
    grip_commands = cm.get_gripper_commands2(pos)
    joint_angles, _ = mc.sort_commands(thetas, grip_commands)
    mc.run(joint_angles)

def calculate_H_matrix_ideal():
    """Calculates H from ideal points, some work is dont to map ideal to real since not all ideal points are measured."""
    # As user which file is ideal
    message = "Select full version of planned path"
    file_prefix, suffix = user_file_select(dirs.PLANNED_PATHS, message, '*_path_*')
    
    #Load ideal points
    name_planned_full = file_prefix+"_path_"+suffix+'.npy'
    file_planned_full = Path(dirs.PLANNED_PATHS, name_planned_full)
    pts_planned_full = np.load(file_planned_full)

    message = "Select ideal path"
    file_prefix, suffix = user_file_select(dirs.PLANNED_PATHS, message, '*_path_*')
    
    #Load ideal points
    name_ideal = file_prefix+"_path_"+suffix+'.npy'
    file_ideal = Path(dirs.PLANNED_PATHS, name_ideal)
    pts_ideal = np.load(file_ideal)

    # Ask user which file to use for real points
    message = "Select measured files to calculate compensation for:"
    file_prefix, suffix = user_file_select(dirs.CAL_TRACKING_DATA_PATH, message, '*_measured*')
    
    # load real and planned points
    name_real = file_prefix+"_measured.npy"
    file_real = Path(dirs.CAL_TRACKING_DATA_PATH, name_real)
    pts_real = np.load(file_real)

    name_planned = file_prefix+"_planned_path.npy"
    file_planned = Path(dirs.CAL_TRACKING_DATA_PATH, name_planned)
    pts_planned = np.load(file_planned)
    
    # filter out the between planned 
    pts_ideal = analyze_transform.filter_unused_ideal_pts(pts_ideal, pts_planned, pts_planned_full)

    H = correction_transform.attempt_minimize_quad(pts_ideal, pts_real)
    print(f"Saving as {file_prefix}_H_matrix{suffix}")

    H_path = Path(dirs.H_MATRIX_PATH, file_prefix+"_H_matrix"+suffix+'.csv')
    correction_transform.save_transformation_matrix(H_path, H)

def calculate_H_matrix_planned():
    """Calculates H from planned points corresponding to measured points. Planned points must be equal to ideal pts"""
    
    # Ask user which file to use for real points
    message = "Select file for Transformation matrix"
    file_prefix, suffix = user_file_select(dirs.CAL_TRACKING_DATA_PATH, message, '*_measured*')
    
    # load real and planned points
    name_real = file_prefix+"_measured.npy"
    file_real = Path(dirs.CAL_TRACKING_DATA_PATH, name_real)
    pts_real = np.load(file_real)

    name_planned = file_prefix+"_planned_path.npy"
    file_planned = Path(dirs.CAL_TRACKING_DATA_PATH, name_planned)
    pts_planned = np.load(file_planned)
    
    H = correction_transform.attempt_minimize_quad(pts_planned, pts_real)
    print(f"Saving as {file_prefix}_H_matrix{suffix}")

    correction_transform.save_transformation_matrix(file_prefix+"_H_matrix"+suffix+'.csv', H)

def generate_transformed_pattern():
    """
    Generate a transformed calibration pattern
    """
    message = "Select transformation Matrix"
    file_prefixes, suffixes = user_file_select_multiple(dirs.H_MATRIX_PATH, message, '*_H_matrix*')
    paths = [Path(dirs.H_MATRIX_PATH, f"{p}_H_matrix{s}.csv") for p, s in zip(file_prefixes, suffixes)]
    H_list = correction_transform.load_transformation_matrix_multiple(paths)

    # change between coordinate systems
    message="\nWhich base path would you like to transform?"
    ideal_prefix, ideal_suffix = user_file_select(dirs.PLANNED_PATHS, message,"*_path_ideal*")
    pts_ideal = np.load(Path(dirs.PLANNED_PATHS, f"{ideal_prefix}_path_ideal{ideal_suffix}.npy"))

    # get the current time for a prefix
    prefix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    name_joint_angles_transformed = prefix+"_ja_transformed_2x"+ideal_suffix
    name_path_transformed = prefix+"_path_transformed_2x"+ideal_suffix
    
    compensated_points = correction_transform.project_points_quad_multiple(pts_ideal, H_list)

    # solve inverse kinematics
    logging.info("solving inverse kinematics...")
    thetas = cm.inverse_kinematics(compensated_points) # convert to joint angles
    grip_commands = cm.get_gripper_commands2(compensated_points) # remove unnecessary wrist commands, add gripper open close instead
    joint_angles, exceeds_lim = mc.sort_commands(thetas, grip_commands)

    logging.info("Correcting joint angles")
    joint_angles, compensated_points = mc.correct_limits(joint_angles, compensated_points, exceeds_lim)
    logging.info("Solved inverse kinetatics. Saving...")

    # if platform.system() == "Windows":
    #     cm.plot_robot(thetas, compensated_points)

    np.save(Path(dirs.PLANNED_PATHS, name_path_transformed), compensated_points)
    np.save(Path(dirs.PLANNED_PATHS, name_joint_angles_transformed), joint_angles)

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
    print('4. Calculate Transformation Matrix (Planned)')
    print('5. Calculate Transformation Matrix (Ideal)')
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
        calculate_H_matrix_planned()
    elif choice == '5':
        calculate_H_matrix_ideal()
    elif choice == "0":
        print("Exiting")
        exit()
    else:
        print("Invalid option")

if __name__ == "__main__":
    log_level = sys.argv[1] if len(sys.argv) > 1 else "Debug"

    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % log_level)
    logging.basicConfig(level=logging.DEBUG)

    user_menu()
    # main()
    # old_main()