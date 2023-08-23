import numpy as np
from IK_Solvers.traditional import ChessMoves
from motor_commands import MotorCommands
from Data_analytics import correction_transform

def string_to_array(s: str) -> np.ndarray:
    # Split the string into a list of strings
    str_list = s.split(',')
    
    # Convert the strings to integers
    int_list = [int(s.strip()) for s in str_list]
    
    # Convert the list of integers to a NumPy array
    int_array = np.array(int_list)
    
    return int_array

def get_gripper_commands_new(waypoints):
    """3rd attempt: manually makes the last step close the gripper"""
    commands = [np.pi/4, 3*np.pi/4] # angles needed for open and closed (in radians)

    grip_commands = np.ones_like(waypoints[0,:]) * commands[0]
    grip_commands[-1] = commands[1]
    return grip_commands

def main():
    cm = ChessMoves(lift=200)
    mc = MotorCommands()
    cur_pos = np.array([0,100,500])

    # ==================Using transformation matrix============
    # load transformation matrix
    print("finding transform")
    H, T, real_mean = correction_transform.get_transform("positions_day2.npy", "path_big_day2.npy")

    def generate_path(start, goal):
        print("Generating quintic line.")
        path = cm.quintic_line(start, goal, 5) # generate waypoints
        path = np.hstack((path, path[:,-1].reshape((3,1))))
        
        print("Updating points")
        # change between coordinate systems
        path_optitrack_sys = correction_transform.to_optitrack_sys(path)
        projected_points = correction_transform.project_points(path_optitrack_sys, real_mean, T, H)
        projected_points = correction_transform.from_optitrack_sys(projected_points)

        print("Inverse Kinematics")
        thetas = cm.inverse_kinematics(projected_points) # convert to joint angles
        grip_commands = get_gripper_commands_new(path) # remove unnecessary wrist commands, add gripper open close instead
        
        cm.plot_robot(thetas, projected_points)
        return mc.sort_commands(thetas, grip_commands)

    while 1:
        value = input("Enter new target:")
        if value == "HOME":
            next_pos = cm.HOME
        else:
            next_pos = cm.get_coords(value)

        print(f"next_pos: {next_pos}")
        path_optitrack_sys = correction_transform.to_optitrack_sys(np.array(next_pos).reshape((3,1)))
        projected_points = correction_transform.project_points(path_optitrack_sys, real_mean, T, H)
        projected_points = correction_transform.from_optitrack_sys(projected_points)
        print(f"Corrected Pos: {projected_points.reshape((1,3))}")
                
        path = generate_path(cur_pos, next_pos)
        print(path[:, -5:])

        input("Are you ready.")
        mc.run(path)

        cur_pos = next_pos
        

if __name__ == "__main__":
    main()