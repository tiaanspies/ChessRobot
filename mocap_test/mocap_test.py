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

def get_filename_planned(search_path:Path):
    """
    USES PATH SPECIFIED BY "CAL_DATA_PATH"
    A function that checks the contents of the "Arm Cal Data" file, 
    prints a list to the user and lets them select a file. Then returns the 
    selected file name
    """
    file_name_generator = search_path.glob("*_path_*")
    file_name_list = [file_name for file_name in file_name_generator]

    # print the list of files
    print("Select a file:")
    for i, file_name in enumerate(file_name_list):
        print(f"{i}: {file_name.name}")

    # get user input
    user_input = input("Enter a number: ")
    user_input = int(user_input)

    # return the selected file name
    name = file_name_list[user_input].stem
    name = name.split("_path_")[0]  # remove everything after and including "_ja_"
    
    return name # remove "_measured" from the end

def run_and_track(tracker: Aruco.ArucoTracker, cam, cal_path: Path):
    # load path
    selected_name = get_filename_planned(dirs.PLANNED_PATHS)

    run_type = 0 # 0: unknown, 1: ideal 2: transformed

    # find the file that has the matching datetime
    # check whether it is ideal or transformed
    for file_name in dirs.PLANNED_PATHS.glob(f"*{selected_name}*"):
        if "path" in file_name.name:
            plan = np.load(file_name)

            if "transformed" in file_name.name:
                run_type = 2
            elif "ideal" in file_name.name:
                run_type = 1

        elif "ja" in file_name.name:
            logging.info(f"Loading {file_name.name}")
            angles = np.load(file_name)
   
    mc.load_path(angles, plan)

    # Initialize tracking variables
    measured = np.zeros((3,0))
    planned_path = np.zeros((3,0))

    # read first pos
    ccs_start_pos = tracker.take_photo_and_estimate_pose(cam)
    
    # rcs_home_pos = np.array([[0], [230], [500]]) # home pos
    logging.debug(f"Home pos: {cm.HOME}")
    rcs_home_pos = cm.HOME.reshape((3,1))
    
    logging.debug(f"Start pos [CCS]: {ccs_start_pos}")

    T_ccs_to_rcs = ccs_start_pos-rcs_home_pos

    # step through
    run_cal = True 
    while run_cal:
        run_cal, plan_points = mc.run_once()

        sleep(2)
        # get position
        ccs_current_pos = tracker.take_photo_and_estimate_pose(cam)

        if ccs_current_pos is not None and plan_points is not None:
            rcs_new_pos = ccs_current_pos-T_ccs_to_rcs

            measured = np.hstack([measured, rcs_new_pos])
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
    H, T, pts_ideal_mean, pts_real_mean = correction_transform.attempt_minimize_quad(pts_ideal, pts_real)
    # H, T, real_mean = correction_transform.get_transform(name_real, name_ideal)
    print("Updating points")

    # change between coordinate systems
    ideal_datetime = analyze_transform.get_filename(path=dirs.PLANNED_PATHS, 
                                                message="\nWhich base path would you like to transform?",
                                                identifier="_path_ideal")
    name_joint_angles_transformed = ideal_datetime+"_ja_transformed"
    name_path_transformed = ideal_datetime+"_path_transformed"
    
    pts_ideal = np.load(Path(dirs.PLANNED_PATHS, f"{ideal_datetime}_path_ideal.npy"))
    projected_points = correction_transform.project_points_quad(pts_ideal, pts_real_mean, T, H)

    # print to check they match
    if platform.system() == "Windows":
        fig = go.Figure()
        analyze_transform.plot_3data(projected_points, fig, "projected")

        # Show the plot
        fig.show()

    # solve inverse kinematics
    print("solving inverse kinematics...")

    # Add 10 copies of the first point to the array.
    # This is to ensure that the robot starts at an untransformed position
    # since the first position "zeros" its coordinates
    home_2_trans_home =  cm.quintic_line(pts_ideal[:,0], projected_points[:,0], 10)

    projected_points = np.hstack((pts_ideal[:,[0]], home_2_trans_home, projected_points))
    thetas = cm.inverse_kinematics(projected_points) # convert to joint angles
    grip_commands = cm.get_gripper_commands2(projected_points) # remove unnecessary wrist commands, add gripper open close instead
    joint_angles = mc.sort_commands(thetas, grip_commands)
    print("solved!")

    # if platform.system() == "Windows":
    #     cm.plot_robot(thetas, projected_points)

    np.save(Path(dirs.PLANNED_PATHS, name_path_transformed), projected_points)
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