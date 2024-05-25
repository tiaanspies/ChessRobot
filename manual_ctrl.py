import chess_manager

def manual_control_menu():
    print("Enter start and end positions of the piece you want to move.")

    start_pos = input('Enter start position: ')
    end_pos = input('Enter end position: ')

    chess_manager.robotsPhysicalMove(start_pos+end_pos, None)


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