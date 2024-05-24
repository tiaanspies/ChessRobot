import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ModuleNotFoundError:
    print("analyze_transform: Did not load plotly, will not plot")
from Data_analytics import correction_transform
from pathlib import Path
import path_directories as dirs
import sys

def main(): 
    if sys.platform == "linux":
        print("This function is not supported on linux as it requires plotly")
        sys.exit()

    # check whether the path is with or without transformation
    reply = input(
        "Select option:\n"\
        "1: Recorded run with transformation\n"\
        "2: Recorded run without transformation\n"\
        "3: Projected run\n"\
        "4: Plot Planned\n"\
        "5: Transformation on transformed points\n"\
        "6: Analyze transformed results\n"
        )
    
    if reply == '1':
        recorded_with_transformation()
    elif reply == '2':
        recorded_without_transformation()
    elif reply == '3':
        projected_run()
    elif reply == '4':
        plot_planned()
    elif reply == '5':
        transformation_on_transformed_pts()     
    elif reply == '6':
        analyze_transformed_results()
    else:
        print("Invalid input")
        sys.exit()       

def create_plot_canvas(a='3D Scatter Plot'):
    fig = go.Figure()
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
    #         xaxis_range=[-150, 150],
    #         yaxis_range=[50, 350],
    #         zaxis_range=[0, 500],
        ),
        title=a
    )
    return fig

def plot_planned():
    message = "Select measured file: "
    date, suffix, ext = get_matching_file_name(dirs.PATH_WIN_PLANNED_PATHS, message, "*_path_*")

    name_planned = date+"_path_"+suffix+ext

    # Load the numpy files for current and actual positions
    pts_planned = np.load(Path(dirs.PATH_WIN_PLANNED_PATHS, name_planned))

    fig = create_plot_canvas(name_planned)
    plot_3data(pts_planned, fig, "Planned")

    # Show the plot
    fig.show()

def recorded_with_transformation():
    message = "Select measured file: "
    date, suffix, ext = get_matching_file_name(dirs.PATH_WIN_CAL_TRACKING_DATA, message, "*_measured*")

    name_real = date+"_measured"+suffix+ext
    name_planned = date+"_planned_path"+suffix+ext

    # Load the numpy files for current and actual positions
    pts_real = np.load(Path(dirs.PATH_WIN_CAL_TRACKING_DATA, name_real))
    pts_planned = np.load(Path(dirs.PATH_WIN_CAL_TRACKING_DATA, name_planned))

    message="\nSelect ideal file: "
    prefix, suffix, ext= get_matching_file_name(dirs.PATH_WIN_PLANNED_PATHS, message, "*_path_ideal*")

    file_ideal = Path(dirs.PATH_WIN_PLANNED_PATHS, prefix+"_path_ideal"+suffix+ext)
    pts_ideal = np.load(file_ideal)

    fig = create_plot_canvas(name_real)
    plot_3data(pts_real, fig, "Pts_real")
    plot_3data(pts_ideal, fig, "pts_ideal")
    plot_3data(pts_planned, fig, "Planned")

    # Show the plot
    fig.show()

def analyze_transformed_results():
    # Get the tracked data points and their planned pts
    message="Select measured file: "
    date, suffix, ext = get_matching_file_name(dirs.PATH_WIN_CAL_TRACKING_DATA, message, identifier="*_measured*")

    # Measured points
    name_real = date+"_measured"+suffix+ext
    pts_real = np.load(Path(dirs.PATH_WIN_CAL_TRACKING_DATA, name_real))

    # planned points
    name_planned = date+"_planned_path"+suffix+ext
    pts_planned = np.load(Path(dirs.PATH_WIN_CAL_TRACKING_DATA, name_planned))

    # Ask user which planned path file was used as the target
    message = "Select full version of planned path"
    file_prefix, suffix, ext = get_matching_file_name(dirs.PATH_WIN_PLANNED_PATHS, message, '*_path_*')
    
    #Load planned points
    name_planned_full = file_prefix+"_path_"+suffix+'.npy'
    pts_planned_full = np.load(Path(dirs.PATH_WIN_PLANNED_PATHS, name_planned_full))

    # get the original untransformed ideal path
    message = "What is the untransformed ideal path"
    ideal_prefix, suffix, ext = get_matching_file_name(dirs.PATH_WIN_PLANNED_PATHS, message,"*_path_ideal*")
    
    name_base = ideal_prefix+"_path_ideal"+suffix+ext
    pts_ideal = np.load(Path(dirs.PATH_WIN_PLANNED_PATHS, name_base))

    # Remove ideal points that were not captured by measurements
    pts_ideal = filter_unused_ideal_pts(pts_ideal, pts_planned, pts_planned_full)

    # Print results
    print(f"\nMSD Error between ideal and projected: {find_msd_error(pts_ideal, pts_real)}")
    print(f"Max Error between ideal and projected: {find_max_error(pts_ideal, pts_real)}")
    print(f"Mean Error between ideal and projected: {find_mean_error(pts_ideal, pts_real)}")

    plot_euclid_error_hist(pts_ideal, pts_real)
    plot_individual_error_hist(pts_ideal, pts_real)

    fig = create_plot_canvas(name_real)
    plot_3data(pts_real, fig, "Pts_real")
    plot_3data(pts_ideal, fig, "pts_ideal")
    fig.show()

def transformation_on_transformed_pts():
    """Applies transformation on measured points and compares with ideal points."""

    # Get the tracked data points and their planned pts
    message="Select measured file: "
    date, suffix, ext = get_matching_file_name(dirs.PATH_WIN_CAL_TRACKING_DATA, message, identifier="*_measured*")

    # Measured points
    name_real = date+"_measured"+suffix+ext
    pts_real = np.load(Path(dirs.PATH_WIN_CAL_TRACKING_DATA, name_real))

    message = "Which Transformation matrix?"
    file_prefix, suffix, ext = get_matching_file_name(dirs.PATH_WIN_H_MATRIX_PATH, message, "*_H_matrix*")
    
    H_path = Path(dirs.PATH_WIN_H_MATRIX_PATH, file_prefix+'_H_matrix'+suffix+ext)
    H = correction_transform.load_transformation_matrix(H_path)

    # get the ideal path
    message = "What is the ideal path"
    ideal_prefix, suffix, ext = get_matching_file_name(dirs.PATH_WIN_PLANNED_PATHS, message,"*_path_ideal*")
    
    name_base = ideal_prefix+"_path_ideal"+suffix+ext
    pts_ideal = np.load(Path(dirs.PATH_WIN_PLANNED_PATHS, name_base))

    pts_projected = correction_transform.project_points_quad(pts_real, H)

    # print transformation matrix and translation
    # correction_transform.print_transform(H, T, pts_real_mean, pts_ideal_mean)
    # print(f"Transformation matrix:\n{H}")
    # print(f"\nMSD Error between ideal and projected: {find_msd_error(pts_ideal, pts_projected)}")
    # print(f"Max Error between ideal and projected: {find_max_error(pts_ideal, pts_projected)}")
    # print(f"Mean Error between ideal and projected: {find_mean_error(pts_ideal, pts_projected)}")

    # plot_euclid_error_hist(pts_ideal, pts_projected)

    fig = create_plot_canvas(name_real)
    plot_3data(pts_real, fig, "Pts_real")
    plot_3data(pts_ideal, fig, "pts_ideal")
    plot_3data(pts_projected, fig, "Projected")
    
    # Show the plot
    fig.show()
    
def recorded_without_transformation():
    message="Select measured file: "
    date, suffix, ext = get_matching_file_name(dirs.PATH_WIN_CAL_TRACKING_DATA, message, identifier="*_measured*")

    name_real = date+"_measured"+suffix+ext
    file_real = Path(dirs.PATH_WIN_CAL_TRACKING_DATA, name_real) 

    # Load the numpy files for current and actual positions
    pts_real = np.load(file_real)

    file_ideal = Path(dirs.PATH_WIN_CAL_TRACKING_DATA, date+"_planned_path"+suffix+ext)
    pts_ideal = np.load(file_ideal)

    #remove points that contain NAN
    mask = ~np.isnan(pts_real).any(axis=0)
    pts_ideal = pts_ideal[:, mask]
    pts_real = pts_real[:, mask]

    H = correction_transform.attempt_minimize_quad(pts_ideal, pts_real)
    pts_projected = correction_transform.project_points_quad(pts_real, H)

    # print transformation matrix and translation
    # correction_transform.print_transform(H, T, pts_real_mean, pts_ideal_mean)
    print(f"Transformation matrix:\n{H}")
    print(f"\nMSD Error between ideal and projected: {find_msd_error(pts_ideal, pts_projected)}")
    print(f"Max Error between ideal and projected: {find_max_error(pts_ideal, pts_projected)}")
    print(f"Mean Error between ideal and projected: {find_mean_error(pts_ideal, pts_projected)}")

    plot_euclid_error_hist(pts_ideal, pts_projected)

    fig = create_plot_canvas(name_real)
    plot_3data(pts_real, fig, "Pts_real")
    plot_3data(pts_ideal, fig, "pts_ideal")
    plot_3data(pts_projected, fig, "Projected")
    
    # Show the plot
    fig.show()

def filter_unused_ideal_pts(pts_ideal, pts_planned, pts_planned_full):
    """Filter out the ideal points that are not used in the planned path"""
    
    assert pts_ideal.shape == pts_planned_full.shape, f"pts_ideal {pts_ideal.shape} "\
        f"and pts_planned_full {pts_planned_full.shape} must have the same shape"
    
    pts_planned_full_copy = pts_planned_full.copy()
    pts_ideal_copy = pts_ideal.copy()

    for i, pt in enumerate(pts_planned.T):
        if not np.array_equal(pt, pts_planned_full_copy.T[i]):
            pts_planned_full_copy = np.delete(pts_planned_full_copy, i, axis=1)
            pts_ideal_copy = np.delete(pts_ideal_copy, i, axis=1)

    if pts_planned_full_copy.shape[1] - pts_planned.shape[1] == 1:
        pts_planned_full_copy = np.delete(pts_planned_full_copy, pts_planned_full_copy.shape[1] - 1, axis=1)
        pts_ideal_copy = np.delete(pts_ideal_copy, pts_planned_full_copy.shape[1] - 1, axis=1)

    assert pts_planned.shape[1] == pts_ideal_copy.shape[1], f"pts_planned {pts_planned.shape[1]} "\
        f"and pts_ideal_copy {pts_ideal_copy.shape[1]} must have the same number of points"

    return pts_ideal_copy

def projected_run():
    message = "Which Transformation matrix?"
    file_prefix, suffix, ext = get_matching_file_name(dirs.PATH_WIN_H_MATRIX_PATH, message, "*_H_matrix*")
    
    H_path = Path(dirs.PATH_WIN_H_MATRIX_PATH, file_prefix+'_H_matrix'+suffix+ext)
    H = correction_transform.load_transformation_matrix(H_path)
    
    message = "Which base path would you like to transform?"
    ideal_prefix, suffix, ext = get_matching_file_name(dirs.PATH_WIN_PLANNED_PATHS, message,"*_path_ideal*")
    
    name_base = ideal_prefix+"_path_ideal"+suffix+ext
    pts_ideal = np.load(Path(dirs.PATH_WIN_PLANNED_PATHS, name_base))
    projected_points = correction_transform.project_points_quad(pts_ideal, H)

    # print to check they match
    fig = create_plot_canvas(name_base)
    plot_3data(pts_ideal, fig, "Ideal")
    plot_3data(projected_points, fig, "Projected")
    

    # Show the plot
    fig.show()

def find_msd_error(pts_a, pts_b):
    """Find the mean squared error between all the points in pts_a and pts_b"""

    # Calculate the mean squared error
    msd = np.mean(np.sum((pts_a-pts_b)**2, axis=0))

    return msd

def find_max_error(pts_truth, pts_test):
    """ Find the max error between truth and test in euclidean distance"""

    # Calculate the max error
    max_error = np.max(np.sqrt(np.sum((pts_truth-pts_test)**2, axis=0)))

    return max_error

def find_mean_error(pts_truth, pts_test):
    """ Find the mean error between truth and test in euclidean distance"""

    # Calculate the mean error
    mean_error = np.mean(np.sqrt(np.sum((pts_truth-pts_test)**2, axis=0)))

    return mean_error

def plot_euclid_error_hist(pts_truth, pts_test):
    """Plot a histogram of the error between the truth and test points"""

    # Calculate the error
    error = np.sqrt(np.sum((pts_truth-pts_test)**2, axis=0))

    # Create a histogram
    # Create a histogram
    fig = go.Figure(data=[go.Histogram(x=error)])
    fig.update_xaxes(title_text='Position Error (mm)')
    fig.update_layout(
        xaxis=dict(
            dtick=0.5
        )
    )
    fig.show()

def plot_individual_error_hist(pts_truth, pts_test):
    """Plot a histogram for the difference between point and target for each axis seperate"""

    # Calculate the error
    error = pts_truth - pts_test

    # Create a histogram
    fig = make_subplots(rows=3, cols=1, subplot_titles=("X position error (mm)", "Y position error (mm)", "Z position error (mm)"))
    for i in range(3):
        fig.add_trace(go.Histogram(x=error[i, :]), row=i+1, col=1)
    fig.show()

def get_matching_file_name(search_path:Path, message:str="Select a file: ", identifier:str="*_path_*"):
    """
    asks the user to select a file from any path that contains '*_path_*'.
    returns the prefix and suffix of the file name.
    """
    file_name_generator = search_path.glob(f"{identifier}")
    file_name_list = sorted([file_name for file_name in file_name_generator], reverse=True)

    # print the list of files
    print("\n"+message)
    for i, file_name in enumerate(file_name_list):
        print(f"{i}: {file_name.name}")

    # get user input
    user_input = input("Enter a number: ")
    user_input = int(user_input)

    # return the selected file name
    name = file_name_list[user_input].stem
    ext = file_name_list[user_input].suffix

    prefix = name.split(identifier.strip("*"))[0]  # remove everything after and including identifier
    suffix = name.split(identifier.strip("*"))[-1]  # remove everything before and including identifier

    return prefix, suffix, ext

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

if __name__ == "__main__":
    main()
