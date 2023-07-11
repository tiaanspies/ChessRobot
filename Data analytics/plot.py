import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the numpy file
data = np.load('positions 4.npy')
data_target = np.load('path_bigXs.npy').T

data_target[:, [0, 1, 2]] = data_target[:, [1, 2, 0]]

def plot_3data(data, ax):
    # Extract x, y, and z coordinates
    x = data[:, 0]
    y = data[:, 2]
    z = data[:, 1]

    ax.scatter(x, y, z)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Scatter Plot')
    ax.set_xlim([0, 500])  # Replace xmin and xmax with your desired limits
    ax.set_ylim([-400, 400])  # Replace ymin and ymax with your desired limits
    ax.set_zlim([0, 600])

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plot_3data(data, ax)
plot_3data(data_target, ax)

# Show the plot
plt.show()