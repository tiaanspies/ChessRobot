import numpy as np
from scipy.optimize import minimize
import path_directories as dirs
from pathlib import Path

def objective_function_quad(X:np.array, pts_ideal:np.array, pts_real:np.array):
    """
    Projects points using X matrix and computes cost of error.

    Projection is in the following form:

        [x'] = [1  2  3  4  5  7  8  9  10 11][x^2 -----------------]
        [y'] = [12 13 14 15 16 17 18 19 20 21][y^2 -----------------]
        [z'] = [22 23 24 25 26 27 28 29 30 31][z^2 -----------------]
                                              [xy ------------------]
                                              [xz ------------------]
                                              [yz ------------------]
                                              [x -------------------]
                                              [y -------------------]
                                              [z -------------------]
                                              [1 -------------------]
    Last element should always be one, therefore elements are divided by the last row in diff calc.
    Can try to modify so that minimization leaves last row of A matrix as [ 0 0 0 1].

    returns cost related to error. 

    TODO: Investigate different cost functions.
    """
    A = X.reshape((3,10))

    # Remove columns where z values are larger than 210

    pts_quad = np.square(pts_real)
    # project points
    xy = pts_real[0, :] * pts_real[1, :]
    xz = pts_real[0, :] * pts_real[2, :]
    yz = pts_real[1, :] * pts_real[2, :]
    pts_transformed = A @ np.vstack([pts_quad[:3, :], xy, xz, yz, pts_real])

    diff = pts_ideal - pts_transformed

    # score using l2 norm
    # dist = np.linalg.norm(diff, axis=1)

    # add squared distance to each point. 
    total = np.sum(diff**2)

    return total

def apply_compensation(pts_ideal:np.array, H_list:np.array, range_dict:dict):
    """Apply transformation matrix to points"""
    pts_transformed = project_points_quad(pts_ideal, H_list)

    min_x = range_dict["min_x"]
    max_x = range_dict["max_x"]
    min_y = range_dict["min_y"]
    max_y = range_dict["max_y"]
    min_z = range_dict["min_z"]
    max_z = range_dict["max_z"]

    # Create mask for points that are within the range
    mask_within_range =  (pts_transformed[0, :] > min_x) & (pts_transformed[0, :] < max_x) & \
            (pts_transformed[1, :] > min_y) & (pts_transformed[1, :] < max_y) & \
            (pts_transformed[2, :] > min_z) & (pts_transformed[2, :] < max_z)
    
    # pts_compensated are pts_transformed where pts_ideal is within range and pts_ideal when pts_ideal is outside range
    pts_compensated = np.copy(pts_ideal)
    pts_compensated[:, mask_within_range] = pts_transformed[:, mask_within_range]

    return pts_compensated

def project_points_quad_multiple(pts:np.array, transformation_matrices:list[np.array]):
    """Apply multiple transformations in series to points"""
    for transformation_matrix in transformation_matrices:
        pts = project_points_quad(pts, transformation_matrix)

    return pts

def project_points_quad(pts:np.array, transformation_matrix:np.array):
    """
    Inputs points that will be projected.

    Points are inputted with each point using a row (shape: 3xn). 
    However the dot product requires them to be cols (shape: nxn).
    They are switched for calculation and then returned to the original shape.

    pts_mean: mean of points that was used to calculate transformation matrix.
    T: translation to move projected points onto target points. 
        (Is usually difference between target and starting point means)
    transformation_matrix: Used to project the zero mean points onto the zero mean target points.
    """
    assert pts.shape[0] == 3, "Points must be in shape (3, n)"

    # Homogeneous coordinates
    ones_row = np.ones((1, pts.shape[1]))

    pts_padded = np.vstack((pts, ones_row))
    xy = pts[0, :] * pts[1, :]
    xz = pts[0, :] * pts[2, :]
    yz = pts[1, :] * pts[2, :]
    pts_quad = np.square(pts[:3, :])
    # project points
    pts_transformed = transformation_matrix @ np.vstack((pts_quad,xy,xz,yz, pts_padded))

    return pts_transformed

def attempt_minimize_quad(pts_ideal:np.array, pts_real:np.array):
    """
 
    """
    # Remove columns where pts_real contains np.nan
    # mask = ~np.isnan(pts_real).any(axis=0)
    # pts_ideal = pts_ideal[:, mask]
    # pts_real = pts_real[:, mask]
    
    assert pts_ideal.shape[0] == 3, "Points must be in shape (3, n)"
    assert pts_real.shape[0] == 3, "Points must be in shape (3, n)"
    pts_ideal_mean = pts_ideal.mean(axis=1, keepdims=True)
    pts_real_mean = pts_real.mean(axis=1, keepdims=True)

    T = pts_ideal_mean-pts_real_mean

    # change points into generalized form
    pt_count = pts_real.shape[1]
    ones_row = np.ones((1, pt_count))

    pts_real_padded = np.vstack([pts_real, ones_row])

    # set points as args
    args = (pts_ideal, pts_real_padded)

    # initialize matrix
    init = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 
                     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]], dtype=np.float64)
    
    init[:, -1] = T[:, 0]
    init = init.reshape((-1))
    tolerance = 1e-3 * pt_count
    # minimize
    res = minimize(
            objective_function_quad, 
            x0=init, args=args,
            # tol=tolerance,
            tol=1e-6,
            options={'maxiter':10000},
            # method="Nelder-Mead" //Gives poor result
            method="BFGS"
            # method="CG"
        )

    print(f"Iterations used: {res.nit}")
            
    H = res.x.reshape((3,10))

    # if res.success == False:
    #     raise ValueError("Unable to minimize for transformation matrix")
    return H

def get_transform(filename_real, filename_ideal):
    """
    Reads file and computes transformation matrix.
    """
    # Load the numpy files for current and actual positions

    pts_real = np.load(Path(dirs.CAL_TRACKING_DATA_PATH, filename_real))
    pts_ideal = np.load(Path(dirs.CAL_TRACKING_DATA_PATH,filename_ideal))

    H = attempt_minimize_quad(pts_ideal, pts_real)

    return H

def save_transformation_matrix(path:str, H:np.array):
    """Saves the transformation matrix in a csv file"""
    np.savetxt(path, H, delimiter=",")

def load_transformation_matrix(path):
    """Loads the transformation matrix from a csv file"""
    return np.loadtxt(path, delimiter=",")

def load_transformation_matrix_multiple(paths):
    """Loads the transformation matrix from a csv file"""
    matrices = [np.loadtxt(path, delimiter=",") for path in paths]

    return matrices

if __name__ == "__main__":
    # main()
    pass


