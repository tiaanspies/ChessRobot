from IK_Solvers.traditional import ChessMoves
from motor_commands import MotorCommands
import numpy as np

cm = ChessMoves(lift=200)
mc = MotorCommands()

def generate_path():
    start = np.array([-200,300,60])
    goal = np.array([200,300,60])
    path = cm.generate_quintic_path(start, goal, None) # generate waypoints
    thetas = cm.inverse_kinematics(path) # convert to joint angles
    grip_commands = cm.get_gripper_commands2(path) # remove unnecessary wrist commands, add gripper open close instead
    return mc.sort_commands(thetas, grip_commands)

def show_feed():

    return None

def main():
    #path = generate_path()
    #np.save("path.npy",path)
    forw_path = np.load("path.npy")
    rev_path = np.flip(forw_path,axis=1)
    forward = True
    
    while True:
        if forward:
            path = forw_path
        else:
            path = rev_path

        forward = not forward
        mc.run(path)
    
if __name__ == "__main__":
    main()