import logging
import sys
import numpy as np
import path_directories as dirs
from Positioning import calibrate_position_compensation as pos_cal
import Positioning.robot_manager
from Chessboard_detection import pi_debugging as debug

def manual_control_menu():
    print('Please select an option:')
    print('1. Point-to-point move')
    print('2. Move to home position')
    print('3. Move to home position backward')
    print('4. Move to home position forward')
    print('5. Take Image')

    choice = input('Enter your choice: ')

    if choice == '1':
        pt2pt_move()
    elif choice == '2':
        move_to_home()
    elif choice == '3':
        move_to_home_backward()
    elif choice == '4':
        move_to_home_forward()
    elif choice == '5':
        take_image()

def move_to_home():
    robot = Positioning.robot_manager.Robot()

    robot.move_home()

def move_to_home_backward():
    robot = Positioning.robot_manager.Robot()

    robot.move_home_backward()

def move_to_home_forward():
    robot = Positioning.robot_manager.Robot()

    robot.move_home_forward()

def take_image():
    robot = Positioning.robot_manager.Robot()
    _, image = robot.cam.read()

    debug.saveTempImg(image, "manual_saved.jpg")

def pt2pt_move():
    # importing here to prevent camera being init in chess_manager class
    robot = Positioning.robot_manager.Robot()
    print("Enter start and end positions of the piece you want to move. e.g. a1e5")

    pos = input('Enter positions: ')

    robot.execute_chess_move(pos, None)

def main():
    log_level = sys.argv[1] if len(sys.argv) > 1 else "Debug"

    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % log_level)
    logging.basicConfig(level=logging.DEBUG)
    
    manual_control_menu()

if __name__ == '__main__':
    main()
