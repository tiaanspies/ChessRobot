from IK_Solvers.traditional import ChessMoves
from motor_commands import MotorCommands
import numpy as np
import matplotlib.pyplot as plt
from Data_analytics import correction_transform
from time import sleep
from pathlib import Path
from Chessboard_detection import Camera_Manager
from Chessboard_detection import Aruco
import datetime
import cv2.aruco as aruco
import path_directories as dirs
from Data_analytics import analyze_transform
import platform
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
    aruco_obj = Aruco.ArucoTracker()

    # generate new pattern and save
    # aruco_obj.load_marker_pattern_positions(12, 17, 35, 26)
    aruco_obj.load_marker_pattern_positions(22, 30, 20, 15)

    return aruco_obj

def get_filename_planned():
    """
    USES PATH SPECIFIED BY "CAL_DATA_PATH"
    A function that checks the contents of the "Arm Cal Data" file, 
    prints a list to the user and lets them select a file. Then returns the 
    selected file name
    """
    file_name_generator = dirs.PLANNED_PATHS.glob("*_path_ideal.npy")
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
    
    return name[:-len("_path_ideal")] # remove "_measured" from the end

def run_and_track(tracker: Aruco.ArucoTracker, cam, cal_path: Path):
    # load path
    selected_name = get_filename_planned()
    plan = np.load(Path(dirs.PLANNED_PATHS, selected_name+"_path_ideal.npy"))
    angles = np.load(Path(dirs.PLANNED_PATHS, selected_name+"_ja_ideal.npy"))
    mc.load_path(angles, plan)

    # Initialize tracking variables
    measured = np.zeros((3,0))
    planned_path_actual = np.zeros((3,0))

    # read first pos
    start_pos = tracker.take_photo_and_estimate_pose(cam)
    home_pos = np.array([[0], [230], [500]]) # home pos
    
    print(f"Start_pos: {start_pos}")

    # step through
    run_cal = True 
    while run_cal:
        run_cal, plan_points = mc.run_once()

        sleep(2)
        # get position
        tvecs = tracker.take_photo_and_estimate_pose(cam)

        if tvecs is not None and plan_points is not None:
            new_pos = tvecs-start_pos+home_pos

            measured = np.hstack([measured, new_pos])
            planned_path_actual = np.hstack([planned_path_actual, plan_points])

            print(f"Position: {new_pos.reshape(1,3)}")
        sleep(1)

    # save data
    prefix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    np.save(Path(cal_path, prefix + "_measured.npy"), measured)
    np.save(Path(cal_path, prefix + "_planned_path.npy"), planned_path_actual)

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

def generate_ideal_pattern(plot_pattern=False):
    """
    Generate an ideal calibration pattern
    """
    vertices = {
    "top" : 250,
    "bottom" : 120,
    "right" : 120,
    "left" : -120,
    "close" : 120,
    "far" : 350}

    # generate waypoints
    path = draw_cube(vertices, 4) 

    # create name
    name = "_".join([str(v) for v in vertices.values()])
    name_path = name+"_path_ideal"
    name_joint_angles = name+"_ja_ideal"

    np.save(Path(dirs.PLANNED_PATHS, name_path), path)
    print("path saved")

    if plot_pattern:
        ax = plt.axes(projection='3d')
        ax.scatter(path[0,:], path[1,:], path[2,:])
        plt.show()    

    print("solving inverse kinematics...")
    thetas = cm.inverse_kinematics(path) # convert to joint angles
    grip_commands = cm.get_gripper_commands2(path) # remove unnecessary wrist commands, add gripper open close instead
    joint_angles = mc.sort_commands(thetas, grip_commands)
    print("solved!")
    # cm.plot_robot(thetas, path)

    np.save(Path(dirs.PLANNED_PATHS, name_joint_angles), joint_angles)

def generate_transformed_pattern():
    """
    Generate a transformed calibration pattern
    """

    file_prefix = analyze_transform.get_filename()
    name_real = file_prefix+"_measured.npy"
    name_ideal = file_prefix+"_planned_path.npy"

    name_joint_angles_transformed = file_prefix+"_ja_transformed"
    name_path_transformed = file_prefix+"_path_transformed"

    # load transformation matrix
    print("Finding transform")
    H, T, real_mean = correction_transform.get_transform(name_real, name_ideal)
    print("Updating points")

    # change between coordinate systems
    pts_ideal = np.load(Path(dirs.CAL_TRACKING_DATA_PATH, name_ideal))
    path_optitrack_sys = correction_transform.to_optitrack_sys(pts_ideal)
    projected_points = correction_transform.project_points(path_optitrack_sys, real_mean, T, H)
    projected_points = correction_transform.from_optitrack_sys(projected_points)

    # print to check they match
    if platform.system() == "Windows":
        fig = go.Figure()
        analyze_transform.plot_3data(projected_points.T[:, [0, 2, 1]], fig, "projected")

        # Set labels and title
    # Set labels and title
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            xaxis_range=[-150, 150],
            yaxis_range=[50, 350],
            zaxis_range=[0, 600],
        ),
        title='3D Scatter Plot'
    )

    # Show the plot
    fig.show()

    # solve inverse kinematics
    print("solving inverse kinematics...")
    thetas = cm.inverse_kinematics(projected_points) # convert to joint angles
    grip_commands = cm.get_gripper_commands2(projected_points) # remove unnecessary wrist commands, add gripper open close instead
    joint_angles = mc.sort_commands(thetas, grip_commands)
    print("solved!")

    cm.plot_robot(thetas, projected_points)

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

def old_main():
    vertices = {
    "top" : 340,
    "bottom" : 120,
    "right" : 120,
    "left" : -120,
    "close" : 120,
    "far" : 520}

    path = draw_cube(vertices, 4) # generate waypoints
    # np.save("mocap_test/path_big_day2.npy", path) # CHANGE THIS SO YOU DON'T OVERWRITE PREVIOUS!
    # print("path generated")
    
    # ax = plt.axes(projection='3d')
    # ax.scatter(path[0,:], path[1,:], path[2,:])
    # # plt.show()

    # print("solving inverse kinematics...")
    # thetas = cm.inverse_kinematics(path) # convert to joint angles
    # grip_commands = cm.get_gripper_commands2(path) # remove unnecessary wrist commands, add gripper open close instead
    # plan = mc.sort_commands(thetas, grip_commands)
    # print("solved!")
    # cm.plot_robot(thetas, path)

    # np.save("mocap_test/plan_big_z.npy",plan)
    # plan = np.load("Data_analytics/plan_big_z.npy",)

    # mc.run(plan)

    # ==================Using transformation matrix============
    # load transformation matrix
    print("finding transform")
    H, T, real_mean = correction_transform.get_transform("positions_day2.npy", "path_big_day2.npy")
    print("Updating points")

    # change between coordinate systems
    path_optitrack_sys = correction_transform.to_optitrack_sys(path)
    projected_points = correction_transform.project_points(path_optitrack_sys, real_mean, T, H)
    projected_points = correction_transform.from_optitrack_sys(projected_points)

    # print to check they match    
    ax = plt.axes(projection='3d')
    ax.scatter(projected_points[0,:], projected_points[1,:], projected_points[2,:])
    plt.show()

    # solve inverse kinematics
    print("solving inverse kinematics...")
    thetas = cm.inverse_kinematics(projected_points) # convert to joint angles
    grip_commands = cm.get_gripper_commands2(projected_points) # remove unnecessary wrist commands, add gripper open close instead
    plan = mc.sort_commands(thetas, grip_commands)
    print("solved!")

    cm.plot_robot(thetas, projected_points)

    np.save("mocap_test/plan_day3_2.npy",plan)
    plan = np.load("mocap_test/plan_day3_2.npy")
    
    mc.run(plan)

if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=2)
    user_menu()
    # main()
    # old_main()