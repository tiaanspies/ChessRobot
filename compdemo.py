from IK_Solvers.traditional import ChessMoves
from motor_commands import MotorCommands
import numpy as np

cm = ChessMoves()
mc = MotorCommands()

def generate_path():
    start = np.array([-200,100,0])
    goal = np.array([200,100,0])
    path = cm.generate_quintic_path(start, goal, None) # generate waypoints
    thetas = cm.inverse_kinematics(path) # convert to joint angles
    grip_commands = cm.get_gripper_commands(path) # remove unnecessary wrist commands, add gripper open close instead
    return mc.sort_commands(thetas, grip_commands)

def main():

    plan = generate_path()
    np.save("path.npy",path)
    mc.run(path)

if __name__ == "__main__":
    main()