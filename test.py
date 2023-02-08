import numpy as np
from IK_Solvers.traditional import ChessMoves
import matplotlib.pyplot as plt
# from motor_commands import MotorCommands

def wrap2pi(angles):
    return angles % (2 * np.pi)

# create class instances
cm = ChessMoves()
# mc = MotorCommands()

# edit the DH parameters if needed
''' Looks like motors move as expected, but solution flips the last joint and both elbow and base angles are too high
also strange discontinuity in first two base angles
l1_params = [0, 3*np.pi/2, 0, 83.35]
l2_params = [0, np.pi, -296, 0]
l3_params = [0, 0, -284.76, 0]
l4_params = [np.pi,0,-90,0]
''' 
'''Looks like motors move as expected, SOLUTION LOOKS CORRECT, but both elbow and base angles are too high
also strange discontinuity in first two base angles
l1_params = [0, 3*np.pi/2, 0, 83.35]
l2_params = [0, np.pi, -296, 0]
l3_params = [0, 0, -284.76, 0]
l4_params = [np.pi,0,90,0]
'''
''' Looks like motors move as expected, but solution flips the last joint and both elbow and base angles are too high
also strange discontinuity in first two base angles
l1_params = [0, 3*np.pi/2, 0, 83.35]
l2_params = [0, np.pi, -296, 0]
l3_params = [0, 0, -284.76, 0]
l4_params = [0,0,-90,0]
'''
'''
# Shoulder rotates down, elbow rotates up, solution puts elbow angle into board, base and elbow too high, shoulder angle jumps across 2pi discontinuity
l1_params = [0, np.pi/2, 0, 83.35]
l2_params = [0, np.pi, -296, 0]
l3_params = [0, 0, -284.76, 0]
l4_params = [np.pi,0,90,0]
'''

# Shoulder and elbow rotate down, SOLUTION LOOKS CORRECT, base and shoulder angles are too high
l1_params = [0, np.pi/2, 0, 83.35]
l2_params = [0, 0, -296, 0]
l3_params = [0, 0, -284.76, 0]
l4_params = [np.pi,0,90,0]
'''
# home to the right of board, shoulder and elbow rotate up, SOLUTION LOOKS CORRECT, base angle too high
l1_params = [0, np.pi/2, 0, 83.35]
l2_params = [0, 0, 296, 0]
l3_params = [0, 0, 284.76, 0]
l4_params = [np.pi,0,90,0]
''''''
# home to the right of board, shoulder rotates down and elbow up, SOLUTION LOOKS CORRECT, base angle too high
l1_params = [0, 3*np.pi/2, 0, 83.35]
l2_params = [0, np.pi, 296, 0]
l3_params = [0, 0, 284.76, 0]
l4_params = [np.pi,0,90,0]
'''
param_list = [l1_params, l2_params, l3_params, l4_params]
cm.initialize_arm(param_list)


# test what direction the motors are moving
'''
thetas = np.zeros((4,1))
while thetas[0] <= .6:
    cm.plot_robot(thetas)
    thetas = thetas + np.pi/12
'''

# set some preliminary positions
home = cm.forward_kinematics(np.zeros((4,1))).reshape((3,))
# start = np.array([0, 0, 300])
# goal = np.array([0, 300, 20])
start = cm.get_coords('a7')
goal = cm.get_coords('e3')
step_len = 10 # mm

# generate a path between them
# path = cm.quintic_line(start, goal, step_len)
path = cm.generate_quintic_path(start, goal)
thetas = wrap2pi(cm.inverse_kinematics(path))

# simulate the robot following that path
'''cm.plot_robot(thetas, path)'''

# now convert the theta commands to work for the real robot
motor_thetas = cm.add_gripper_commands(thetas)
motor_thetas[0,:] = ((2*np.pi - motor_thetas[0,:]) - np.pi/4) * 2 # fix the base angle by switching rot direction, shifting to the front slice, then handling the gear ratio
motor_thetas[1,:] = 2*np.pi - motor_thetas[1,:] # make any necessary changes to the shoulder angles
motor_thetas[2,:] = motor_thetas[2,:] + np.pi # make any necessary changes to the elbow angles
motor_thetas = wrap2pi(motor_thetas) # confirm one more time that all angles are as low as possible

# plot all the angles to confirm they are between 0 - pi
steps = len(motor_thetas[0,:])
plt.plot(np.linspace(0,steps,steps), motor_thetas[0,:], color='r',label="base")
plt.plot(np.linspace(0,steps,steps), motor_thetas[1,:], color='b',label="shoulder")
plt.plot(np.linspace(0,steps,steps), motor_thetas[2,:], color='y',label="elbow")
plt.legend()
plt.show() 

if any(motor_thetas.ravel() > np.pi):
    raise ValueError('IK solution requires angles greater than the 180-degree limits of motors')
# run the commands on the physical robot
# mc.run(motor_thetas)

