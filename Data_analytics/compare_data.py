import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the numpy file
# data = np.load('positions 4.npy')
# data_target = np.load('path_bigXs.npy').T

data = np.load('positions.npy')[165:, :]
data_target = np.load("Data_analytics/plan_big_z.npy").T

data_target[:, [0, 1, 2]] = data_target[:, [1, 2, 0]]
data_target = data_target[data_target[:, 1] == 100]

# Create a figure and a grid of subplots
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

for axis in axes:
    axis.axis('equal')
    # axis.set_aspect('equal', 'box')

pass

def plot_3data(axes, data):
    # Plot data on each subplot
    axes[0].plot(data[:, 0], data[:, 1])
    axes[0].set_title('Plot 1')

    axes[1].plot(data[:, 1], data[:, 2])
    axes[1].set_title('Plot 2')

    axes[2].plot(data[:, 0], data[:, 2])
    axes[2].set_title('Plot 3')

plot_3data(axes, data)
plot_3data(axes, data_target)
plt.show()