from IK_Solvers.traditional import ChessMoves
from motor_commands import MotorCommands
import numpy as np
import matplotlib.pyplot as plt
from Data_analytics import correction_transform

cm = ChessMoves()
mc = MotorCommands()

def fake_inverse_kinematics(path):
    return np.vstack((path,np.zeros_like(path[0,:])))

def draw_cube(v, slice_num):
    top_left = np.array([v["left"],v["close"],v["top"]])
    bottom_left = np.array([v["left"],v["close"],v["bottom"]])
    top_right = np.array([v["right"],v["close"],v["top"]])
    bottom_right = np.array([v["right"],v["close"],v["bottom"]])
    step = 20
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
        
def main():

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
    # plan = np.load("mocap_test/plan_big_z.npy",)
    
    # mc.run(plan)

    # ==================Corrected values ============
    H, T, real_mean = correction_transform.get_transform("positions_day2.npy", "path_big_day2.npy")
    path_optitrack_sys = correction_transform.to_optitrack_sys(path)
    projected_points = correction_transform.project_points(path_optitrack_sys, real_mean, T, H)
    projected_points = correction_transform.from_optitrack_sys(projected_points)

    ax = plt.axes(projection='3d')
    ax.scatter(projected_points[0,:], projected_points[1,:], projected_points[2,:])
    plt.show()

    print("solving inverse kinematics...")
    thetas = cm.inverse_kinematics(projected_points) # convert to joint angles
    grip_commands = cm.get_gripper_commands2(projected_points) # remove unnecessary wrist commands, add gripper open close instead
    plan = mc.sort_commands(thetas, grip_commands)
    print("solved!")
    cm.plot_robot(thetas, projected_points)

    np.save("mocap_test/plan_day3_1.npy",plan)
    # plan = np.load("mocap_test/plan_big_z.npy",)
    
    # mc.run(plan)


if __name__ == "__main__":
    main()