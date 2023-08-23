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
        
        print("Updating points")
        # change between coordinate systems
        path_optitrack_sys = correction_transform.to_optitrack_sys(path)
        projected_points = correction_transform.project_points(path_optitrack_sys, real_mean, T, H)
        projected_points = correction_transform.from_optitrack_sys(projected_points)

        print("inverse inematics")
        thetas = cm.inverse_kinematics(projected_points) # convert to joint angles
        grip_commands = cm.get_gripper_commands2(path) # remove unnecessary wrist commands, add gripper open close instead

        
        cm.plot_robot(thetas, projected_points)
        return mc.sort_commands(thetas, grip_commands)

    while 1:
        value = input("Enter new target:")
        next_pos = cm.get_coords(value)
        print(f"next_pos: {next_pos}")
        
        path = generate_path(cur_pos, next_pos)

        mc.run(path)

        cur_pos = next_pos
        

if __name__ == "__main__":
    main()