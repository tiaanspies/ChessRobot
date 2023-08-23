from IK_Solvers.traditional import ChessMoves
from motor_commands import MotorCommands
from Data_analytics import correction_transform
import numpy as np

cm = ChessMoves(lift=200)
mc = MotorCommands()

# ==================Using transformation matrix============
# load transformation matrix
print("finding transform")
H, T, real_mean = correction_transform.get_transform("positions_day2.npy", "path_big_day2.npy")

def generate_path():
    pickup_square = input("Pick up square:")
    start = cm.get_coords(pickup_square)
    drop_square = input("Drop off square:")
    goal = cm.get_coords(drop_square)

    path = cm.generate_quintic_path(start, goal, None) # generate waypoints
    
    print("Updating points")
    # change between coordinate systems
    path_optitrack_sys = correction_transform.to_optitrack_sys(path)
    projected_points = correction_transform.project_points(path_optitrack_sys, real_mean, T, H)
    projected_points = correction_transform.from_optitrack_sys(projected_points)

    print("Inverse Kinematics")
    thetas = cm.inverse_kinematics(path) # convert to joint angles
    grip_commands = cm.get_gripper_commands2(path) # remove unnecessary wrist commands, add gripper open close instead
    return mc.sort_commands(thetas, grip_commands)

def show_feed():

    return None

def main():
    path = generate_path()
    np.save("path.npy",path)
    forw_path = np.load("path.npy")
    rev_path = np.flip(forw_path,axis=1)
    forward = True
    
    while True:
        if forward:
            path = forw_path
        else:
            path = rev_path

        forward = not forward
        input("ready?")
        mc.run(path)
    
if __name__ == "__main__":
    main()