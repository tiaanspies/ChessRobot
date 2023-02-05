import numpy as np
from IK_Solvers.traditional import ChessMoves
import matplotlib.pyplot as plt

def wrap2pi(angles):
    return angles % (2 * np.pi)

cm = ChessMoves()
start = cm.get_coords('a2')
goal = cm.get_coords('g6')
path = cm.generate_quintic_path(start, goal, None)
thetas1 = wrap2pi(cm.inverse_kinematics(path))
path1 = cm.forward_kinematics(thetas1)
cm.plot_robot(thetas1, path1)

steps = len(thetas1[0,:])
plt.plot(np.linspace(0,steps,steps), 2*np.pi-thetas1[0,:], color='r',label="base")
plt.plot(np.linspace(0,steps,steps), thetas1[1,:], color='b',label="shoulder")
plt.plot(np.linspace(0,steps,steps), thetas1[2,:], color='y',label="elbow")
plt.plot(np.linspace(0,steps,steps), thetas1[3,:], color='k',label="wrist")
plt.legend()
plt.show() 

#cm.plot_robot(thetas1, path1)


thetas = np.zeros((4,1))
while thetas[0] <= .6:
    cm.plot_robot(thetas)
    thetas = thetas + np.pi/12

