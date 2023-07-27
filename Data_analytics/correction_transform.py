import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize

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

def to_optitrack_sys(pts_ideal):
    return (pts_ideal.T)[:, [1, 2, 0]]

def from_optitrack_sys(pts):
    return (pts[:, [2, 0, 1]]).T


def get_transform(filename_real, filename_ideal):
    # Load the numpy files for current and actual positions
    try:
        prefix = "Data_analytics\\"
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
