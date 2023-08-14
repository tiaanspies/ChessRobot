from IK_Solvers.traditional import ChessMoves
from motor_commands import MotorCommands
import numpy as np
import matplotlib.pyplot as plt
from Data_analytics import correction_transform
from time import sleep
from pathlib import Path
from Chessboard_detection import Camera_Manager
from Chessboard_detection import Aruco

cm = ChessMoves()
mc = MotorCommands()



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
    # define aruco pattern file location.
    name = "7x5_small_3x2_large"
    file_path_and_name = Path("Chessboard_detection", "Aruco Markers", name).resolve().__str__()

    # create aruco tracker object
    aruco_obj = Aruco.ArucoTracker()

    # generate new pattern and save
    aruco_obj.generate_and_save_marker_pattern(file_path_and_name)

    # load pattern
    # correction = actual length from top of pattern to bottom / predicted
    size_correction = 186.5/191.29374999999996
    aruco_obj.load_marker_pattern(file_path_and_name, size_correction)

    return aruco_obj

def run_and_track(tracker, cam):
    # load path
    plan = np.load("Data_analytics/plan_big_z.npy",)
    mc.load_path(plan)

    # Initialize tracking variables
    positions = np.zeros((3,0))

    # read first pos
    rvecs, start_pos = tracker.estimate_pose(cam)
    home_pos = np.array([[0], [230], [500]]) # come pos

    # step through
    while mc.run_once():
        sleep(2)
        # get position
        rvecs, tvecs = tracker.estimate_pose(cam)

        if tvecs is None:
            new_pos = np.zeros((3,1))
        else:
            new_pos = tvecs-start_pos+home_pos

        positions = np.hstack([positions, new_pos])
        sleep(1)

    np.save("positions.npy", positions)

def save_path():
   
    vertices = {
    "top" : 340,
    "bottom" : 120,
    "right" : 120,
    "left" : -120,
    "close" : 120,
    "far" : 520}

    path = draw_cube(vertices, 4) # generate waypoints
    # np.save("mocap_test/path_big_day2.npy", path) # CHANGE THIS SO YOU DON'T OVERWRITE PREVIOUS!
    print("path generated")
    
    ax = plt.axes(projection='3d')
    ax.scatter(path[0,:], path[1,:], path[2,:])
    plt.show()

    print("solving inverse kinematics...")
    thetas = cm.inverse_kinematics(path) # convert to joint angles
    grip_commands = cm.get_gripper_commands2(path) # remove unnecessary wrist commands, add gripper open close instead
    plan = mc.sort_commands(thetas, grip_commands)
    print("solved!")
    cm.plot_robot(thetas, path)

    np.save("data_analytics/plan_big_z.npy",plan)

def main():
    save_path()

    # # load aruco obj things
    # aruco_obj = create_tracker()

    # # create camera object
    # cam = Camera_Manager.RPiCamera(loadSavedFirst=False)

    # run_and_track(aruco_obj, cam)

def old_main():
        vertices = {
        "top" : 340,
        "bottom" : 120,
        "right" : 120,
        "left" : -120,
        "close" : 120,
        "far" : 520}

    # path = draw_cube(vertices, 4) # generate waypoints
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
    # print("finding transform")
    # H, T, real_mean = correction_transform.get_transform("positions_day2.npy", "path_big_day2.npy")
    # print("Updating points")
    # path_optitrack_sys = correction_transform.to_optitrack_sys(path)
    # projected_points = correction_transform.project_points(path_optitrack_sys, real_mean, T, H)
    # projected_points = correction_transform.from_optitrack_sys(projected_points)
    
    # ax = plt.axes(projection='3d')
    # ax.scatter(projected_points[0,:], projected_points[1,:], projected_points[2,:])
    # plt.show()

    # print("solving inverse kinematics...")
    # thetas = cm.inverse_kinematics(projected_points) # convert to joint angles
    # grip_commands = cm.get_gripper_commands2(projected_points) # remove unnecessary wrist commands, add gripper open close instead
    # plan = mc.sort_commands(thetas, grip_commands)
    # print("solved!")

    # cm.plot_robot(thetas, projected_points)

    # np.save("mocap_test/plan_day3_2.npy",plan)
    # plan = np.load("mocap_test/plan_day3_2.npy")
    
    # mc.run(plan)

if __name__ == "__main__":
    main()