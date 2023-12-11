import numpy as np

try:
    import plotly.graph_objects as go
except ModuleNotFoundError:
    print("analyze_transform: Did not load plotly, will not plot")
from scipy.optimize import minimize
from Data_analytics import correction_transform
from pathlib import Path
import path_directories as dirs
import sys

def main():
    if sys.platform == "linux":
        print("This function is not supported on linux as it requires plotly")
        sys.exit()

    # declare global variables
    global pts_real_subset
    global pts_ideal_subset

    # name_real = "positions_pi_cam2.npy"
    # name_real = "positions_day2.npy"
    # name_ideal = "path_big_day2.npy"
    date = get_filename()

    name_real = date+"_measured.npy"
    name_ideal = date+"_planned_path.npy"

    file_real = Path(dirs.CAL_TRACKING_DATA_PATH, name_real)
    file_ideal = Path(dirs.CAL_TRACKING_DATA_PATH, name_ideal)

    # Load the numpy files for current and actual positions
    # rearrange to print nicely
    pts_real = np.load(file_real).T[:, [0, 2, 1]]
    pts_ideal = (np.load(file_ideal).T)[:, [0, 2, 1]]

    H, T, pts_ideal_mean, pts_real_mean = correction_transform.attempt_minimize(pts_ideal, pts_real)
    
    # ===================================================================
    # Create a 3D scatter plot
    fig = go.Figure()

    # plot target positions
    plot_3data(pts_ideal, fig, "pts_ideal")

    # find predictions and plot
    plot_3data(project_points(pts_real, pts_real_mean, T, H), fig, "Projected")
    # plot_3data(pts_real, fig, "Pts_real")

    # Set labels and title
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            xaxis_range=[-150, 150],
            yaxis_range=[50, 350],
            zaxis_range=[0, 600],
        ),
        title='3D Scatter Plot'
    )

    # Show the plot
    fig.show()

def get_filename():
    """
    USES PATH SPECIFIED BY "CAL_DATA_PATH"
    A function that checks the contents of the "Arm Cal Data" file, 
    prints a list to the user and lets them select a file. Then returns the 
    selected file name
    """
    file_name_generator = dirs.CAL_TRACKING_DATA_PATH.glob("*_measured.npy")
    file_name_list = [file_name for file_name in file_name_generator]

    # print the list of files
    print("Select a file:")
    for i, file_name in enumerate(file_name_list):
        print(f"{i}: {file_name.name}")

    # get user input
    user_input = input("Enter a number: ")
    user_input = int(user_input)

    # return the selected file name
    name = file_name_list[user_input].stem
    
    return name[:-len("_measured")] # remove "_measured" from the end

def plot_3data(pts_real, fig, lab):
    # Extract x, y, and z coordinates
    x = pts_real[:, 0]
    y = pts_real[:, 2]
    z = pts_real[:, 1]

    # shapes = ['circle', 'cross', 'diamond', 'square', 'x']
    shapes = ['circle']
    shape = np.random.choice(shapes)

    fig.add_trace(go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=4,
            # color='blue',
            symbol=shape,
            opacity=0.8
        ),
        name=lab
    ))

def objective_function(X:np.array, pts_ideal:np.array, pts_real:np.array):
    # coef = np.vstack([X.reshape((3,4)), [0, 0, 0, 1]])
    A = X.reshape((4,4))
    # [x'] = [1  2  3  4 ][x]
    # [y'] = [5  6  7  8 ][y]
    # [z'] = [9  10 11 12][z]
    # [1'] = [13 14 15 16][1]

    # [x'] = [1 2 3][x]
    # [y'] = [4 5 6][y]
    # [z'] = [7 8 9][z]

    #project points
    pts_transformed = A@pts_real

    # find the difference and normalize using 4th element
    diff = pts_ideal/pts_ideal[3, :] - pts_transformed/pts_transformed[3, :]

    # score using l2 norm, excluding 4th element
    dist = np.linalg.norm(diff[:3, :], axis=0)
    total = np.sum(dist**2)
    return total

def project_points(pts_real:np.array, pts_real_mean:np.array, T:np.array, transformation_matrix:np.array):
    # Homogeneous coordinates
    pts_real_general_0_mean = np.hstack((pts_real-pts_real_mean, np.ones((pts_real.shape[0], 1))))
    
    # # Apply transformation
    projected_points_general_0_mean = np.dot(transformation_matrix ,pts_real_general_0_mean.T)
    
    # Convert back to Cartesian coordinates
    projected_points_0_mean = projected_points_general_0_mean[:3, :] / projected_points_general_0_mean[3:, :]
    
    return projected_points_0_mean.T+pts_real_mean+T

def attempt_minimize(pts_ideal:np.array, pts_real:np.array):
    pts_ideal_mean = pts_ideal.mean(axis=0)
    pts_real_mean = pts_real.mean(axis=0)

    T = pts_ideal_mean-pts_real_mean

    # attempt to use minimize
    ones_col = np.ones((pts_real.shape[0], 1))

    pts_real_zero_mean = np.hstack([pts_real-pts_real_mean, ones_col])
    pts_ideal_zero_mean = np.hstack([pts_ideal-pts_ideal_mean, ones_col])

    # objective_function(a.flatten(), pts_real.T, pts_ideal.T)
    args = (pts_ideal_zero_mean.T, pts_real_zero_mean.T)
    init = np.eye(4).reshape((-1))
    res = minimize(
            objective_function, 
            x0=init, args=args,
            # tol=1e-3,
            # options={'maxiter':10000},
            # method="Nelder-Mead" //Gives poor result
            method="BFGS"
            # method="CG"
        )
            
    print(f"Success: {res.success}")
    print(res)
    H = res.x.reshape((4,4))

    return H, T, pts_ideal_mean, pts_real_mean

if __name__ == "__main__":
    main()
