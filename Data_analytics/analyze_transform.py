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
    
    plots_for_report()
    return

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


def plots_for_report(): 

    # ==================== Recorded with transformation ====================
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

    # remove NAN points
    mask = ~(np.isnan(pts_real).any(axis=0) | np.isnan(pts_ideal).any(axis=0))
    pts_ideal_w = pts_ideal[:, mask]
    pts_real_w = pts_real[:, mask]
    pts_planned = pts_planned[:, mask]

    # ==================== Recorded without transformation ====================
    message="Select measured file: "
    date, suffix, ext = get_matching_file_name(dirs.PATH_WIN_CAL_TRACKING_DATA, message, identifier="*_measured*")

    name_real = date+"_measured"+suffix+ext
    file_real = Path(dirs.PATH_WIN_CAL_TRACKING_DATA, name_real) 

    # Load the numpy files for current and actual positions
    pts_real = np.load(file_real)

    file_ideal = Path(dirs.PATH_WIN_CAL_TRACKING_DATA, date+"_planned_path"+suffix+ext)
    pts_ideal = np.load(file_ideal)

    #remove points that contain NAN
    mask = ~(np.isnan(pts_real).any(axis=0) | np.isnan(pts_ideal).any(axis=0))
    pts_ideal_wo = pts_ideal[:, mask]
    pts_real_wo = pts_real[:, mask]

    H = correction_transform.attempt_minimize_quad(pts_ideal_wo, pts_real_wo)

    print(f"Transformation matrix:\n{H}")
    pts_projected = correction_transform.project_points_quad(pts_real_wo, H)

    plot_comparison_euclid_error_hist(pts_ideal_w, pts_real_w, pts_ideal_wo, pts_projected)
    plot_comparison_individual_error_hist(pts_ideal_w, pts_real_w, pts_ideal_wo, pts_projected)

    fig = create_plot_canvas(name_real)
    plot_3data(pts_real, fig, "Measured Points")
    plot_3data(pts_ideal, fig, "Nominal Points")
    plot_3data(pts_projected, fig, "Projected after compensation")
    
    # Show the plot
    fig.show()
    # ==================== Visualise lin regression ====================
    # Create a line plot
    fig = go.Figure()

    ys = np.linspace(0, 500, 200)

    x = 106.25
    z = 130
    # [x'] = [1  2  3  4  5  7  8  9  10 11][x^2 -----------------]
    # [y'] = [12 13 14 15 16 17 18 19 20 21][y^2 -----------------]
    # [z'] = [22 23 24 25 26 27 28 29 30 31][z^2 -----------------]
    #                                         [xy ------------------]
    #                                         [xz ------------------]
    #                                         [yz ------------------]
    #                                         [x -------------------]
    #                                         [y -------------------]
    #                                         [z -------------------]
    #                                         [1 -------------------]
    zs_comp = x*x*H[2,0] + ys*ys*H[2, 1] + z*z*H[2, 2] + x*ys*H[2, 3] + x*z*H[2, 4] + ys*z*H[2, 5] + \
         x*H[2, 6] + ys*H[2, 7] + z*H[2, 8] + H[2, 9]

    fig.add_trace(go.Scatter(
        x=ys,
        y=[z]*len(ys),
        mode='lines',
        name='Nominal Z value'
    ))

    fig.add_trace(go.Scatter(
        x=ys,
        y=zs_comp,
        mode='lines',
        name='Compensated Z values'
    ))

    mask_x_106_z_159_wo = (pts_ideal_wo[0, :] == x) & (pts_ideal_wo[2, :] == z)
    pts_real_wo_106_159 = pts_real_wo[2, mask_x_106_z_159_wo]

    print(f"Count of matches in mask: {np.sum(mask_x_106_z_159_wo)}")

    
    mask_x_106_z_159_w = (pts_ideal_w[0, :] == x) & (pts_ideal_wo[2, :] == z)
    pts_real_w_106_159 = pts_real_w[2, mask_x_106_z_159_w]
    ys = np.linspace(420, 130, len(pts_real_wo_106_159))
    fig.add_trace(go.Scatter(
        x=ys,
        y=pts_real_wo_106_159,
        mode='markers',
        name='Measured Z values without compensation'
    ))

    fig.add_trace(go.Scatter(
        x=ys,
        y=pts_real_w_106_159,
        mode='markers',
        name='Measured Z values with compensation'
    ))

    # Set y-axis range
    fig.update_yaxes(range=[0, 200])
    fig.update_xaxes(title_text='Y position (mm)')
    fig.update_yaxes(title_text='Z position (mm)')
    fig.update_layout(title='Snippet of Compensation on Z values')

    # Show the plot
    fig.show()



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

    # remove NAN points
    mask = ~(np.isnan(pts_real).any(axis=0) | np.isnan(pts_ideal).any(axis=0))
    pts_ideal = pts_ideal[:, mask]
    pts_real = pts_real[:, mask]
    pts_planned = pts_planned[:, mask]

    fig = create_plot_canvas(name_real)
    plot_3data(pts_real, fig, "Measured Points")
    plot_3data(pts_ideal, fig, "Nominal Points")
    plot_3data(pts_planned, fig, "Compensated Points")

    # Show the plot
    fig.show()

    # print transformation matrix and translation
    # correction_transform.print_transform(H, T, pts_real_mean, pts_ideal_mean)
    # print(f"Transformation matrix:\n{H}")
    print(f"\nMSD Error between ideal and projected: {find_msd_error(pts_ideal, pts_real)}")
    print(f"Max Error between ideal and projected: {find_max_error(pts_ideal, pts_real)}")
    print(f"Mean Error between ideal and projected: {find_mean_error(pts_ideal, pts_real)}")

    plot_euclid_error_hist(pts_ideal, pts_real)
    plot_individual_error_hist(pts_ideal, pts_real)

    

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
    mask = ~(np.isnan(pts_real).any(axis=0) | np.isnan(pts_ideal).any(axis=0))
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

    plot_euclid_error_hist(pts_ideal, pts_real)
    plot_individual_error_hist(pts_ideal, pts_real)

    fig = create_plot_canvas(name_real)
    plot_3data(pts_real, fig, "Measured Points")
    plot_3data(pts_ideal, fig, "Nominal Points")
    plot_3data(pts_projected, fig, "Projected after compensation")
    
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


def user_file_select_multiple(search_path:Path, message:str="Select a file: ", identifier:str="*_path_*"):
    """
    asks the user to select a file from any path that contains '*_path_*'.
    returns the prefix and suffix of the file name.
    """
    file_name_generator = search_path.glob(f"{identifier}")
    file_name_list = sorted([file_name for file_name in file_name_generator], reverse=True)

    # print the list of files
    print(message)
    for i, file_name in enumerate(file_name_list):
        print(f"{i}: {file_name.name}")

    # get user input
    user_input = input("Enter a number or q to continue: ")
    selected_file_prefixes = []
    selected_file_suffixes = []

    while user_input != "q":
        
        user_input = int(user_input)

        # return the selected file name
        name = file_name_list[user_input].stem
        selected_file_prefixes.append(name.split(identifier.strip("*"))[0])  # remove everything after and including "_path_"
        selected_file_suffixes.append(name.split(identifier.strip("*"))[-1])  # remove everything before and including "_path_"

        user_input = input("Enter a number or q to continue: ")

    return selected_file_prefixes, selected_file_suffixes

def projected_run():
    message = "Which Transformation matrix?"
    file_prefixes, suffixes = user_file_select_multiple(dirs.PATH_WIN_H_MATRIX_PATH, message, '*_H_matrix*')
    
    paths = [Path(dirs.PATH_WIN_H_MATRIX_PATH, f"{p}_H_matrix{s}.csv") for p, s in zip(file_prefixes, suffixes)]
    H_list = correction_transform.load_transformation_matrix_multiple(paths)
    
    message = "Which base path would you like to transform?"
    ideal_prefix, suffix, ext = get_matching_file_name(dirs.PATH_WIN_PLANNED_PATHS, message,"*_path_ideal*")
    
    name_base = ideal_prefix+"_path_ideal"+suffix+ext
    pts_ideal = np.load(Path(dirs.PATH_WIN_PLANNED_PATHS, name_base))
    projected_points = correction_transform.project_points_quad_multiple(pts_ideal, H_list)

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

def plot_comparison_individual_error_hist(pts_real_w, pts_test_with_transformation, pts_real_wo, pts_test_without_transformation):
    """Plot histograms comparing the individual axis errors between the truth and test points with and without transformation"""

    # Calculate the error for both with and without transformation
    error_with_transformation = pts_real_w - pts_test_with_transformation
    error_without_transformation = pts_real_wo - pts_test_without_transformation

    # Create a histogram
    fig = make_subplots(rows=3, cols=1)
    fig.update_layout(title_text='Histograms of Individual Axis Error Distributions Comparison')

    axis_labels = ["X position error (mm)", "Y position error (mm)", "Z position error (mm)"]
    colors = ['blue', 'red']

    for i in range(3):
        fig.add_trace(go.Histogram(
            x=error_without_transformation[i, :],
            xbins=dict(size=1),
            opacity=0.5,
            marker_color=colors[1],
            showlegend=(i == 0),
            name='Predicted Error After Calibration'
        ), row=i+1, col=1)

        fig.add_trace(go.Histogram(
            x=error_with_transformation[i, :],
            xbins=dict(size=1),
            opacity=0.5,
            marker_color=colors[0],
            showlegend=(i == 0),
            name='Measured Error After Calibration'
        ), row=i+1, col=1)

        fig.update_yaxes(title_text="Count", row=i+1, col=1)
        fig.update_xaxes(title_text=axis_labels[i], range=[-25, 30], dtick=5, row=i+1, col=1)

    fig.update_layout(barmode='overlay')
    fig.show()

def plot_comparison_euclid_error_hist(pts_truth_w, pts_test_with_transformation, pts_truth_wo, pts_test_without_transformation):
    """Plot a histogram comparing the error between the truth and test points with and without transformation"""

    # Calculate the error for both with and without transformation
    error_with_transformation = np.sqrt(np.sum((pts_truth_w - pts_test_with_transformation)**2, axis=0))
    error_without_transformation = np.sqrt(np.sum((pts_truth_wo - pts_test_without_transformation)**2, axis=0))

    # Filter errors to be less than 40
    error_with_transformation = error_with_transformation[error_with_transformation < 40]
    error_without_transformation = error_without_transformation[error_without_transformation < 40]

    # Create a histogram
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=error_without_transformation,
        xbins=dict(size=1),
        opacity=0.5,
        name='Predicted Error After Calibration',
        marker_color='red'
    ))

    fig.add_trace(go.Histogram(
        x=error_with_transformation,
        xbins=dict(size=1),
        opacity=0.5,
        name='Measured Error After Calibration',
        marker_color='blue'
    ))

    fig.update_xaxes(title_text='Euclidian Position Error (mm)', range=[0, 30])
    fig.update_yaxes(title_text='Count')
    fig.update_layout(title_text='Histogram of Error Distribution Comparison')
    fig.update_layout(
        xaxis=dict(
            dtick=5
        ),
        barmode='overlay'
    )
    fig.show()

def plot_euclid_error_hist(pts_truth, pts_test):
    """Plot a histogram of the error between the truth and test points"""

    # Calculate the error
    error = np.sqrt(np.sum((pts_truth-pts_test)**2, axis=0))
    error = error[error < 40]
    # Create a histogram

    fig = go.Figure(data=[go.Histogram(x=error, xbins=dict(size=1))])
    fig.update_xaxes(title_text='Abs Euclidian Position Error (mm)', range=[0, 30])
    fig.update_yaxes(title_text='Count')
    fig.update_layout(title_text='Histogram of Error Distribution Post Calibration')
    fig.update_layout(
        xaxis=dict(
            dtick=5
        )
    )
    fig.show()

def plot_individual_error_hist(pts_truth, pts_test):
    """Plot a histogram for the difference between point and target for each axis separate"""

    # Calculate the error
    error = pts_truth - pts_test

    # Create a histogram
    fig = make_subplots(rows=3, cols=1)
    fig.update_layout(title_text='Histograms of Individual Axis Error Distributions')

    axis_labels = ["X position error (mm)", "Y position error (mm)", "Z position error (mm)"]

    for i in range(3):
        fig.add_trace(go.Histogram(x=error[i, :], xbins=dict(size=1)), row=i+1, col=1)
        fig.update_yaxes(title_text="Count", row=i+1, col=1)
        fig.update_xaxes(title_text=axis_labels[i], range=[-25, 30], dtick=5, row=i+1, col=1)
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
