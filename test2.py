from IK_Solvers.traditional import ChessMoves
from motor_commands import MotorCommands
import numpy as np
import matplotlib.pyplot as plt

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
    
    slice_width = v["far"] - v["close"] / slice_num
    for _ in range(slice_num):
        path = np.hstack((path, \
                          cm.quintic_line(top_left, top_right, step), \
                          cm.quintic_line(top_right, bottom_left, step), \
                          cm.quintic_line(bottom_left, bottom_right, step), \
                          cm.quintic_line(bottom_right, top_left, step), \
                          cm.quintic_line(top_left, top_left + np.array([0, slice_width, 0]), step)))
        
def main():

    vertices = {
        "top" : 320,
        "bottom" : 20,
        "right" : 160,
        "left" : -160,
        "close" : 20,
        "far" : 320}

    path = cm.draw_cube(vertices, 4) # generate waypoints
    print("path generated")
    plt.plot(path[0,:], path[1,:], path[2,:])

    print("solving inverse kinematics...")
    thetas = cm.inverse_kinematics(path) # convert to joint angles
    grip_commands = cm.get_gripper_commands2(path) # remove unnecessary wrist commands, add gripper open close instead
    plan = mc.sort_commands(thetas, grip_commands)
    print("solved!")

    np.save("plan.npy",plan)
    # plan = np.load("plan.npy",)
    
    # mc.run(plan)

if __name__ == "__main__":
    main()