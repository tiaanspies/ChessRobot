import numpy as np
from IK_Solvers.traditional import ChessMoves
from motor_commands import MotorCommands

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

    def generate_path(start, goal):
        path = cm.quintic_line(start, goal, 5) # generate waypoints
        thetas = cm.inverse_kinematics(path) # convert to joint angles
        grip_commands = cm.get_gripper_commands2(path) # remove unnecessary wrist commands, add gripper open close instead
        return mc.sort_commands(thetas, grip_commands)

    while 1:
        value = input("Enter new target:")
        next_pos = string_to_array(value)

        path = generate_path(cur_pos, next_pos)
        mc.run(path)

        cur_pos = next_pos
        

if __name__ == "__main__":
    main()