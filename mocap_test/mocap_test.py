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

def draw_cube(v, slice_num):
    top_left = np.array([v["left"],v["close"],v["top"]])
    bottom_left = np.array([v["left"],v["close"],v["bottom"]])
    top_right = np.array([v["right"],v["close"],v["top"]])
    bottom_right = np.array([v["right"],v["close"],v["bottom"]])
    step = 10
    path = cm.quintic_line(cm.HOME, top_left, step)
    
    slice_width = (v["far"] - v["close"]) / slice_num
    slice_step = np.array([0.0, slice_width, 0.0], dtype=int)

    for i in range(slice_num):
        path = np.hstack((path, \
                          cm.quintic_line(top_left, bottom_right, step), \
                          cm.quintic_line(bottom_right, bottom_left, step), \
                          cm.quintic_line(bottom_left, top_right, step), \
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
    logging.debug("Creating aruco tracker.")
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
    file_name_list = [file_name for file_name in file_name_generator]

    # print the list of files
    print(message)
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

def run_and_track(tracker: Aruco.ArucoTracker, cam, cal_path: Path):
    # load path
    dimensions, transform_type = user_file_select(dirs.PLANNED_PATHS)

    plan = [f for f in dirs.PLANNED_PATHS.glob(f"{dimensions}_path_{transform_type}.npy")]
    angles = [f for f in dirs.PLANNED_PATHS.glob(f"{dimensions}_ja_{transform_type}.npy")]

    if len(plan) != 1 or len(angles) != 1:
        raise("Multiple or no matching files found. \n Have Tiaan fix his code")
        
    plan = np.load(plan[0])
    angles = np.load(angles[0])
   
    mc.load_path(angles, plan)
    moves_total = angles.shape[1]
    moves_current = 0

    # Initialize tracking variables
    measured = np.zeros((3,0))
    planned_path = np.zeros((3,0))

    # step through
    run_cal = True 
    while run_cal:
        run_cal, plan_points = mc.run_once()
        moves_current += 1

        print(f"Progress: {moves_current/moves_total*100:.2f}%")

        sleep(2)
        # get position
        ccs_current_pos = tracker.take_photo_and_estimate_pose(cam)

        if ccs_current_pos is not None and plan_points is not None:

            measured = np.hstack([measured, ccs_current_pos])
            planned_path = np.hstack([planned_path, plan_points])
            
            logging.debug(f"Position: {ccs_current_pos.reshape(1,3)}")
        sleep(1)

    # save data
    prefix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    logging.info("Saving recorded data.")
    np.save(Path(cal_path, prefix + "_measured.npy"), measured)
    np.save(Path(cal_path, prefix + "_planned_path.npy"), planned_path)

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
    "bottom" : 120,
    "right" : 90,
    "left" : -90,
    "close" : 150,
    "far" : 350}

    # generate waypoints
    logging.info("Generating ideal pattern")
    logging.info(f"Vertices: {vertices}")
    path = draw_cube(vertices, 4) 

    # create name
    name = "_".join([str(v) for v in vertices.values()])
    name_path = name+"_path_ideal"
    name_joint_angles = name+"_ja_ideal"

    logging.info(f"Saving path as '{name_path}'")
    np.save(Path(dirs.PLANNED_PATHS, name_path), path)

    # if plot_pattern:
    #     ax = plt.axes(projection='3d')
    #     ax.scatter(path[0,:], path[1,:], path[2,:])
    #     plt.show()    

    logging.info("solving for joint angles (Inverse Kinematics).")
    thetas = cm.inverse_kinematics(path) # convert to joint angles

    logging.info("Adding gripper commands")
    grip_commands = cm.get_gripper_commands2(path) # remove unnecessary wrist commands, add gripper open close instead
    joint_angles = mc.sort_commands(thetas, grip_commands)
    logging.info(f"Saving joing angles as '{name_joint_angles}'")
    # cm.plot_robot(thetas, path)

    np.save(Path(dirs.PLANNED_PATHS, name_joint_angles), joint_angles)

def generate_transformed_pattern():
    """
    Generate a transformed calibration pattern
    """

    file_prefix = analyze_transform.get_filename(path=dirs.CAL_TRACKING_DATA_PATH)
    name_real = file_prefix+"_measured.npy"
    name_ideal = file_prefix+"_planned_path.npy"

    # load transformation matrix
    print("Finding transform")
    file_real = Path(dirs.CAL_TRACKING_DATA_PATH, name_real) 
    file_ideal = Path(dirs.CAL_TRACKING_DATA_PATH, name_ideal)

    pts_ideal = np.load(file_ideal)
    pts_real = np.load(file_real)

    H = correction_transform.attempt_minimize_quad(pts_ideal, pts_real)
    print("Updating points")

    # change between coordinate systems
    ideal_prefix, ideal_suffix = user_file_select(
        path=dirs.PLANNED_PATHS, 
        message="\nWhich base path would you like to transform?",
        identifier="*_path_ideal*"
    )
    name_joint_angles_transformed = ideal_prefix+"_ja_transformed"+ideal_suffix
    name_path_transformed = ideal_prefix+"_path_transformed"+ideal_suffix
    
    pts_ideal = np.load(Path(dirs.PLANNED_PATHS, f"{ideal_prefix}_path_ideal{ideal_suffix}"))
    compensated_points = correction_transform.project_points_quad(pts_ideal, H)

    # solve inverse kinematics
    print("solving inverse kinematics...")

    # Add 10 copies of the first point to the array.
    # This is to ensure that the robot starts at an untransformed position
    # since the first position "zeros" its coordinates
    home_2_trans_home =  cm.quintic_line(pts_ideal[:,0], compensated_points[:,0], 10)

    compensated_points = np.hstack((pts_ideal[:,[0]], home_2_trans_home, compensated_points))
    thetas = cm.inverse_kinematics(compensated_points) # convert to joint angles
    grip_commands = cm.get_gripper_commands2(compensated_points) # remove unnecessary wrist commands, add gripper open close instead
    joint_angles = mc.sort_commands(thetas, grip_commands)
    print("solved!")

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
    print("4. Exit")

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
    elif choice == "4":
        print("Exiting")
        exit()
    else:
        print("Invalid option")

if __name__ == "__main__":
    log_level = sys.argv[1] if len(sys.argv) > 1 else "Debug"

    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % log_level)
    logging.basicConfig(filename='example.log', filemode='w', level=logging.DEBUG)

    user_menu()
    # main()
    # old_main()