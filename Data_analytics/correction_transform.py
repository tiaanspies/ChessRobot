import numpy as np
from scipy.optimize import minimize

def objective_function(X:np.array, pts_ideal:np.array, pts_real:np.array):
    """
    Projects points using X matrix and computes cost of error.

    Projection is in the following form:

        [x'] = [1  2  3  4 ][x]
        [y'] = [5  6  7  8 ][y]
        [z'] = [9  10 11 12][z]
        [1'] = [13 14 15 16][1]

    Last element should always be one, therefore elements are divided by the last row in diff calc.
    Can try to modify so that minimization leaves last row of A matrix as [ 0 0 0 1].

    TODO: Check that last row (elements 13 - 16 are needed)
    TODO: check for easier cost function instead of l2 norm then squaring in next step.

    returns cost related to error. 
    """
    A = np.vstack([X.reshape((3,4)), [0, 0, 0, 1]])
    # A = X.reshape((4,4))

    #project points
    pts_transformed = A @ pts_real

    # find the difference and normalize using 4th element
    diff = pts_ideal/pts_ideal[3, :] - pts_transformed/pts_transformed[3, :]
    # diff = pts_ideal - pts_transformed

    # score using l2 norm, excluding 4th element
    dist = np.linalg.norm(diff[:3, :], axis=0)

    # add squared distance to each point. 
    total = np.sum(dist**2)

    return total

def project_points(pts:np.array, pts_mean:np.array, T:np.array, transformation_matrix:np.array):
    """
    Inputs points that will be projected.

    Points are inputted with each point using a row (shape: nx3). 
    However the dot product requires them to be cols (shape: 3xn).
    They are switched for calculation and then returned to the original shape.

    pts_mean: mean of points that was used to calculate transformation matrix.
    T: translation to move projected points onto target points. 
        (Is usually difference between target and starting point means)
    transformation_matrix: Used to project the zero mean points onto the zero mean target points.
    """
    # Homogeneous coordinates
    pts_zero_0_mean = pts-pts_mean
    ones_col = np.ones((pts.shape[0], 1))

    pts_general_0_mean = np.hstack((pts_zero_0_mean, ones_col))
    
    # # Apply transformation
    projected_points_general_0_mean = np.dot(transformation_matrix ,pts_general_0_mean.T)
    
    # Convert back to Cartesian coordinates by dividing by last col which should be ones.
    projected_points_0_mean = projected_points_general_0_mean[:3, :] / projected_points_general_0_mean[3:, :]
    
    # translate points.
    # TODO: Simplify by translating directly to target instead of first to original then to target.
    translated_points = projected_points_0_mean.T + pts_mean + T
    return translated_points

def attempt_minimize(pts_ideal:np.array, pts_real:np.array):
    """
    Make points zero mean and save mean to return functions later.  
    
    """
    pts_ideal_mean = pts_ideal.mean(axis=0)
    pts_real_mean = pts_real.mean(axis=0)

    T = pts_ideal_mean-pts_real_mean

    # change points into generalized form
    pt_count = pts_real.shape[0]
    ones_col = np.ones((pt_count, 1))

    pts_real_zero_mean = np.hstack([pts_real-pts_real_mean, ones_col])
    pts_ideal_zero_mean = np.hstack([pts_ideal-pts_ideal_mean, ones_col])

    # set points as args
    args = (pts_ideal_zero_mean.T, pts_real_zero_mean.T)

    # initialize matrix
    init = np.eye(4)[:3,:].reshape((-1))
    tolerance = 1e-3 * pt_count
    # minimize
    res = minimize(
            objective_function, 
            x0=init, args=args,
            tol=tolerance,
            options={'maxiter':10000},
            # method="Nelder-Mead" //Gives poor result
            method="BFGS"
            # method="CG"
        )
            
    # reshape transformation matrix
    # H = res.x.reshape((4,4))
    H = np.vstack([res.x.reshape((3,4)), [0, 0, 0, 1]])

    if res.success == False:
        raise ValueError("Unable to minimize for transformation matrix")

    return H, T, pts_ideal_mean, pts_real_mean

def to_optitrack_sys(pts_ideal):
    """
    Change from coordinate order Noah used to the one used in optitrack 
    """
    return (pts_ideal.T)[:, [1, 2, 0]]

def from_optitrack_sys(pts):
    """
    Change from the optitrack form to the form Noah used.
    """
    return (pts[:, [2, 0, 1]]).T


def get_transform(filename_real, filename_ideal):
    """
    Reads file and computes transformation matrix.
    """
    # Load the numpy files for current and actual positions
    try:
        prefix = "Data_analytics/"
        pts_real = np.load(prefix+filename_real)[:-1, :]
        pts_ideal = to_optitrack_sys(np.load(prefix+filename_ideal))
    except FileNotFoundError:
        prefix = ""
        pts_real = np.load(prefix+filename_real)[:-1, :]
        pts_ideal = to_optitrack_sys(np.load(prefix+filename_ideal))
    
    
    # H, T, pts_ideal_mean, pts_real_mean = attempt_minimize(pts_ideal_subset, pts_real_subset)
    H, T, pts_ideal_mean, pts_real_mean = attempt_minimize(pts_ideal, pts_real)

    return H, T, pts_real_mean
    # ===================================================================
    # projected_points = project_points(pts_real, pts_real_mean, T, H)


if __name__ == "__main__":
    # main()
    pass
