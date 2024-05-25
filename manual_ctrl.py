import chess_manager
import logging

def manual_control_menu():
    print('Please select an option:')
    print('1. Point-to-point move')
    print('2. Move to home position')

    choice = input('Enter your choice: ')

    if choice == '1':
        pt2pt_move()
    elif choice == '2':
        move_to_home()

def pt2pt_move():
    print("Enter start and end positions of the piece you want to move.")

    start_pos = input('Enter start position: ')
    end_pos = input('Enter end position: ')

    chess_manager.robotsPhysicalMove(start_pos+end_pos, None)

def move_to_home():
    """Move to the home position"""
    ja = chess_manager.motion_planner.inverse_kinematics(chess_manager.motion_planner.HOME)
    chess_manager.motor_driver.run(ja)

    logging.info("Moved to home position")

if __name__ == '__main__':
    
    print('Welcome to the Chess Manager!')
    print('Please select an option:')
    print('1. Start a new game')
    print('2. Manual control')

    choice = input('Enter your choice: ')

    if choice == '1':
        pass
    elif choice == '2':
        manual_control_menu()