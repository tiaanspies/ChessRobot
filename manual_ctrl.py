import chess_manager as chess_man
import logging
import sys
import numpy as np
import path_directories as dirs
from mocap_test import mocap_test as mt

def manual_control_menu():
    print('Please select an option:')
    print('1. Point-to-point move')
    print('2. Move to home position')
    print('3. Run previous plan')

    choice = input('Enter your choice: ')

    if choice == '1':
        pt2pt_move()
    elif choice == '2':
        move_to_home()
    elif choice == '3':
        run_previous()

def pt2pt_move():
    print("Enter start and end positions of the piece you want to move.")

    start_pos = input('Enter start position: ')
    end_pos = input('Enter end position: ')

    chess_man.robotsPhysicalMove(start_pos+end_pos, None)

def run_previous():
    prefix, suffix = mt.user_file_select(dirs.RUN_PATH, identifier="*_measured*")
    plan_ja = [f for f in dirs.RUN_PATH.glob(f"*{prefix}_measured{suffix}.npy")]
    
    assert len(plan_ja) == 1, "Multiple or no matching files found. \n Have Tiaan fix his code"

    plan_ja = np.load(plan_ja[0])

    chess_man.motor_driver.run(plan_ja)

def move_to_home():
    """Move to the home position"""
    home_coords = chess_man.motion_planner.HOME
    ja, grip_commands = chess_man.motion_planner.coords_to_joint_angles(home_coords, False)

    chess_man.motor_driver.filter_run(ja, grip_commands)

    logging.info("Moved to home position")

if __name__ == '__main__':

    log_level = sys.argv[1] if len(sys.argv) > 1 else "Debug"

    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % log_level)
    logging.basicConfig(level=logging.DEBUG)
    
    print('Welcome to the Chess Manager!')
    print('Please select an option:')
    print('1. Start a new game')
    print('2. Manual control')

    choice = input('Enter your choice: ')

    if choice == '1':
        pass
    elif choice == '2':
        manual_control_menu()