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
    date = get_filename(path=dirs.CAL_TRACKING_DATA_PATH, message="Select measured file: ", identifier="_measured")
    name_real = date+"_measured.npy"
    file_real = Path(dirs.CAL_TRACKING_DATA_PATH, name_real) 

    # Load the numpy files for current and actual positions
    pts_real = np.load(file_real)

    # Create a 3D scatter plot
    fig = go.Figure()

    # check whether the path is with or without transformation
    transformation = input("\nIs the path with or without transformation? (Enter 1: 'with' or 2:'without'): ")
    if transformation == '1':
        prefix = get_filename(path=dirs.PLANNED_PATHS, message="\nSelect ideal file: ", identifier="_path_ideal")
        file_ideal = Path(dirs.PLANNED_PATHS, prefix+"_path_ideal.npy")
        pts_ideal = np.load(file_ideal)

        plot_3data(pts_real, fig, "Pts_real")
        plot_3data(pts_ideal, fig, "pts_ideal")
    elif transformation == '2':
        prefix = date+"_planned_path.npy"
        file_ideal = Path(dirs.CAL_TRACKING_DATA_PATH, date+"_planned_path.npy")
        pts_ideal = np.load(file_ideal)

        H, T, pts_ideal_mean, pts_real_mean = correction_transform.attempt_minimize_quad(pts_ideal, pts_real)
        
        print(f"H:\n{H}\nT:\n{T}\npts_ideal_mean:\n{pts_ideal_mean}\npts_real_mean:\n{pts_real_mean}")
        # plot_3data(pts_real, fig, "Pts_real")
        plot_3data(correction_transform.project_points_quad(pts_real, pts_real_mean, T, H), fig, "Projected")
        plot_3data(pts_ideal, fig, "pts_ideal")
    else:
        print("Invalid input")
        sys.exit()       

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

def get_filename(path, message="Select a file:", identifier="_measured"):
    """
    USES PATH SPECIFIED BY "CAL_DATA_PATH"
    A function that checks the contents of the "Arm Cal Data" file, 
    prints a list to the user and lets them select a file. Then returns the 
    selected file name
    """
    file_name_generator = path.glob(f"*{identifier}.npy")
    file_name_list = [file_name for file_name in file_name_generator]

    # print the list of files
    print(message)
    for i, file_name in enumerate(file_name_list):
        print(f"{i}: {file_name.name}")

    # get user input
    user_input = input("Enter a number: ")
    user_input = int(user_input)

    # return the selected file name
    name = file_name_list[user_input].stem
    
    return name[:-len(f"{identifier}")] # remove "_measured" from the end

def plot_3data(pts_real, fig, lab):
    # Extract x, y, and z coordinates
    x = pts_real[0, :]
    y = pts_real[1, :]
    z = pts_real[2, :]

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

if __name__ == "__main__":
    main()
