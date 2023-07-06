import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the numpy file
data = np.load('positions 5.npy')

# Extract x, y, and z coordinates
x = data[:, 0]
y = data[:, 2]
z = data[:, 1]

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Scatter Plot')
ax.set_xlim([0, 500])  # Replace xmin and xmax with your desired limits
ax.set_ylim([-400, 400])  # Replace ymin and ymax with your desired limits
ax.set_zlim([0, 600])

# Show the plot
plt.show()