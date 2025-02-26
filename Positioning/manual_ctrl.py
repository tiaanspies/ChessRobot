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
    print('6. Stress test')

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
    elif choice == '6':
        stress_test()

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
    robot = Positioning.robot_manager.Robot()
    print("Enter start and end positions of the piece you want to move. e.g. a1e5")

    pos = input('Enter positions: ')

    robot.execute_chess_move(pos, None)
    
def stress_test():
    """Move the robot to various positions and move pieces around"""

    moves = [
        # move pawns to center rows
        "a2a4",
        "a7a5",
        "b2b4",
        "b7b5",
        "c2c4",
        "c7c5",
        "d2d4",
        "d7d5",
        "e2e4",
        "e7e5",
        "f2f4",
        "f7f5",
        "g2g4",
        "g7g5",
        "h2h4",
        "h7h5",
        # move rook forward, swap other rooks around
        "h8h6",
        "a1h8",
        "a8a1",
        "h1a8",
        "h6h1",
        # move knight forward, swap other knights around
        "g1f3",
        "g8g1",
        "b1g8",
        "b8b1",
        "f3b8",
        # move bishop forward, swap other bishops around
        "f1h3",
        "c8f1",
        "c1c8",
        "f8c1",
        "h3f8",
        # move king and queen forward, swap other around
        "e1e2",
        "d8e1",
        "e2d8",

        "e8e7",
        "d1e8",
        "e7d1",

        # move pawns from center to opposite side
        "a4a7",
        "b4b7",
        "c4c7",
        "d4d7",
        "e4e7",
        "f4f7",
        "g4g7",
        "h4h7",

        "a5a2",
        "b5b2",
        "c5c2",
        "d5d2",
        "e5e2",
        "f5f2",
        "g5g2",
        "h5h2",
    ]

    robot = Positioning.robot_manager.Robot()
    robot.move_home()

    try:
        robot.motor_commands.slow_first_move = False
        for move in moves:
            robot.execute_chess_move(move, None)
    except:
        robot.motor_commands.slow_first_move = True

def main():
    log_level = sys.argv[1] if len(sys.argv) > 1 else "Debug"

    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % log_level)
    logging.basicConfig(level=logging.DEBUG)
    
    manual_control_menu()

if __name__ == '__main__':
    main()
