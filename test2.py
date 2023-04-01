from IK_Solvers.traditional import ChessMoves
from motor_commands import MotorCommands
import numpy as np

cm = ChessMoves()
mc = MotorCommands()

def fake_inverse_kinematics(path):
    return np.vstack((path,np.zeros_like(path[0,:])))

start = np.array([-200,300,60])
goal = np.array([200,300,60])
path = cm.generate_quintic_path(start, goal, None) # generate waypoints
thetas = fake_inverse_kinematics(path) # convert to joint angles
grip_commands = cm.get_gripper_commands2(path) # remove unnecessary wrist commands, add gripper open close instead
plan = mc.sort_commands(thetas, grip_commands)