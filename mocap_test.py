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

cm = ChessMoves()
mc = MotorCommands()

def main():
    save_path()

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

def run_and_track(tracker: Aruco.ArucoTracker, cam, cal_path: Path):
    # load path
    angles = np.load("Data_analytics/plan_big_z2.npy")
    plan = np.load("Data_analytics/plan_big_z_250pts.npy")
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

def save_path():
   
    vertices = {
    "top" : 340,
    "bottom" : 120,
    "right" : 120,
    "left" : -120,
    "close" : 120,
    "far" : 400}

    path = draw_cube(vertices, 4) # generate waypoints
    np.save("Data_analytics/plan_big_z_250pts.npy", path) # CHANGE THIS SO YOU DON'T OVERWRITE PREVIOUS!
    print("path generated")
    
    # ax = plt.axes(projection='3d')
    # ax.scatter(path[0,:], path[1,:], path[2,:])
    # plt.show()

    # print("solving inverse kinematics...")
    # thetas = cm.inverse_kinematics(path) # convert to joint angles
    # grip_commands = cm.get_gripper_commands2(path) # remove unnecessary wrist commands, add gripper open close instead
    # plan = mc.sort_commands(thetas, grip_commands)
    # print("solved!")
    # # cm.plot_robot(thetas, path)

    # np.save("Data_analytics/plan_big_z2.npy",plan)

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
    "top" : 340,
    "bottom" : 120,
    "right" : 120,
    "left" : -120,
    "close" : 120,
    "far" : 400}

    path = draw_cube(vertices, 4) # generate waypoints
    name = "_".join([str(v) for v in vertices.values()])
    np.save(Path(dirs.PLANNED_PATHS, name), path)
    print("path generated")

    if plot_pattern:
        ax = plt.axes(projection='3d')
        ax.scatter(path[0,:], path[1,:], path[2,:])
        plt.show()    

def generate_transformed_pattern():
    """
    Generate a transformed calibration pattern
    """
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

    np.save("Data_analytics/plan_day3_2.npy",plan)

def user_menu():
    """
    Create a user menu with the following options:
    1. Run calibration
    2. generate ideal calibration pattern
    3. generate transformed calibration pattern
    """

    print("1. Run calibration")
    print("2. Generate ideal calibration pattern")
    print("3. Generate transformed calibration pattern")
    print("4. Exit")

    choice = input("Select an option: ")

    if choice == "1":
        print("Running calibration")
        run_calibration()
    elif choice == "2":
        print("Generating ideal calibration pattern")
        generate_ideal_pattern()
    elif choice == "3":
        print("Generating transformed calibration pattern")
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