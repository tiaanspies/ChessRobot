"""
### PSEUDOCODE ###
INITIALIZE
imports libraries, define functions, start stockfish

SETUP GAME
have human pick a side to play
create representations of the board in numpy-chess and in -1, 0, 1 representation for the vision
have the robot go first if it's playing white

loop: when the human makes their turn (hits the timer)
    HUMAN TURN
    use computer vision to represent the board as -1, 0 and 1s
    compare this to the previous to figure out what move was made
    make sure the move is valid
    save the move in the board represenations
    if the human won
        exit

    ROBOT TURN
    determine the best move
    save the move in the board representations
    make the move physically
    if the robot won
        exit
"""

from stockfish import Stockfish
import numpy as np
import chess
import matplotlib.pyplot as plt
from NLinkArm3d import NLinkArm

### INITIALIZE ###

# necessary functions
def whichColor():
    """Human decides which color to play. Black is -1, White is 1"""
    ans = input("Hello, human. Do you want to play black or white today? (b/w): ").strip().lower()
    if ans not in ['b', 'w']:
        print(f'"{ans}" is invalid, please try again...')
        return whichColor()
    elif ans == 'b':
        HUMAN = -1
    elif ans == 'w':
        HUMAN = 1
    ROBOT = -HUMAN
    return HUMAN, ROBOT

def humansTurnFinished():
    """Simple yes/no question to determine if human is done with their turn. This will later be the function that flags when the clock is pressed"""

    # for debugging sake, I currently have this set permanently to True
    return True

    ans = input("Are you finished? (y/n): ").strip().lower()
    if ans not in ['y', 'n']:
        print(f'"{ans}" is invalid, please try again...')
        return humansTurnFinished()
    elif ans == 'n':
        print('Take your time, human')
        return humansTurnFinished()
    elif ans == 'y':
        return True
    return False

def updateVisBoard(board, move, player):
    """Updates the -1, 0, 1 representation of the board"""
    start_square = chess.parse_square(move[:2])
    end_square = chess.parse_square(move[2:])
    board.ravel()[start_square] = 0
    board.ravel()[end_square] = player
    
    return board

def seeBoard(board):
    """Uses CV to determine which squares are occupied, returns -1, 0, 1 representation"""
    # instead right now it takes the users input for which piece to move where
    
    ans = input("what is your move? ")
    board = updateVisBoard(board, ans, HUMAN)

    return board
    
def compareVisBoards(current, previous):
    """compares the CV output with most recent board and outputs the move that was made or None if it can't tell"""
    '''
    # debugging
    print("current:")
    print(current)
    print("previous:")
    print(previous)
    print("compared:")
    print(current!=previous)
    print("Human:")
    print(current==HUMAN)
    '''

    start_square = np.flatnonzero(np.logical_and((current!=previous),(current==0)))
    end_square = np.flatnonzero(np.logical_and((current!=previous),(current==HUMAN)))
    if start_square.size == 0:
        print("failed to locate which piece was moved")
        return None
    elif end_square.size == 0:
        print("failed to locate where the piece was played")
        return None
    
    start_name = chess.square_name(start_square[0])
    end_name = chess.square_name(end_square[0])
    
    return start_name + end_name

def gameOver():
    """returns False if game isn't over, prints the outcome and returns True if it is"""
    if pyboard.outcome():
        print("Good game, human")
        print(pyboard.outcome())     
        return True
    return False

def robotsVirtualMove(visboard):
    """takes in the game in it's current state and returns it having made one best move or None if robot won"""
    
    # ask the engine for the best move
    best_move = stockfish.get_best_move()

    # make that move and update the board representations
    stockfish.make_moves_from_current_position([best_move])
    visboard = updateVisBoard(visboard, best_move, ROBOT)
    pyboard.push_uci(best_move)

    # print the board so the human can make a next move - this won't be needed later
    stockboard = stockfish.get_board_visual(HUMAN == chess.WHITE)
    print(stockboard)

    return best_move, visboard

def perceiveHumanMove():
    """take image (or typed move for now) and return the move that was made"""
    seen_visboard = seeBoard(current_visboard.copy()) # Tiaan's vision function to find occupied squares  
    human_move = compareVisBoards(seen_visboard, current_visboard) # Compare boards to figure out what piece moved
    if human_move is None:
        return perceiveHumanMove()
    return seen_visboard, human_move

def getWaypoints_simple(start, goal, lift=50, N=100):
    """creates a 3xN array of waypoints from the 3x1 start to the 3x1 end"""
    N_up = round(N/3)
    N_over = N - N_up*2
    going_up = np.vstack((np.linspace(start[0], start[0], N_up),
                         np.linspace(start[1], start[1], N_up),
                         np.linspace(start[2], start[2] + lift, N_up)))
    going_over = np.vstack((np.linspace(start[0], goal[0], N_over),
                           np.linspace(start[1], goal[1], N_over),
                           np.linspace(start[2] + lift, goal[2] + lift, N_over)))
    going_down = np.vstack((np.linspace(goal[0], goal[0], N_up),
                           np.linspace(goal[1], goal[1], N_up),
                           np.linspace(goal[2] + lift, goal[2], N_up)))
    waypoints = np.hstack((going_up, going_over, going_down))

    '''
    # for debugging
    print(waypoints)
    ax = plt.axes(projection='3d')
    ax.scatter3D(waypoints[0],waypoints[1],waypoints[2])
    plt.show()
    '''

    return waypoints

def defBoardCoords(square_width=30, border_width=10, base_dist=15, base_height=10):
    """gives an 3x8x8 ndarray representing the euclidean coordinates of the center of each board square"""
    # zero is at the robot base, which is centered base_dist from the edge of the board
    
    # initialize
    board_coords = np.zeros((3,8,8)) 

    # set z coord
    board_coords[2,:,:] = base_height

    # define and set x and y coords
    file_coords = np.linspace(-3.5*square_width,3.5*square_width,8,endpoint=True)
    rank_coords = np.linspace(7.5*square_width,0.5*square_width,8,endpoint=True) + border_width + base_dist
    board_coords[:2,:,:] = np.array(np.meshgrid(file_coords,rank_coords))

    return board_coords

def getCoords(square_name, board_coords):
    """gives real-world coordinates in mm based on chess board square (e.g. 'e2')"""
    file_idx = chess.FILE_NAMES.index(square_name[0]) if ROBOT==1 else chess.FILE_NAMES[::-1].index(square_name[0])
    rank_idx = chess.RANK_NAMES[::-1].index(square_name[1]) if ROBOT==1 else chess.RANK_NAMES.index(square_name[1])
    return board_coords[:,rank_idx,file_idx]

def robotsPhysicalMove(robot_move):
    start = getCoords(robot_move[:2],cboard)
    goal = getCoords(robot_move[2:],cboard)
    coords_path = getWaypoints_simple(start, goal,N=100)
    print(f"start: {start}, goal: {goal}")
    # TODO: find thetapath using inverse kinematics

def defRobotArm(L1=250,L2=250):
    """creates an instance of the NLinkArm class for our specific robot configuration"""
    # define robot parameters in DH convention [theta alpha a d]
    l1_params = [0, np.pi/2, 0, 0]
    l2_params = [0, 0, L1, 0]
    l3_params = [0, 0, L2, 0]
    param_list = [l1_params, l2_params, l3_params]
    return NLinkArm(param_list)


### SETUP GAME ###

# have human pick which side to play (1 = white, -1 = black)
HUMAN, ROBOT = whichColor()

# create an instance of of the stockfish engine with the parameters requested
stockfish = Stockfish(r"C:\Users\HP\Documents\Chess Robot\stockfish\stockfish_15_win_x64_popcnt\stockfish_15_x64_popcnt.exe", depth=15, parameters={"UCI_Elo":800})

# create NLinkArm instance specific to our robot using python_robotics NLinkArm class
robotarm = defRobotArm()

# Define the -1, 0, 1 (visboard), python-chess (pyboard), and coordinate (cboard) representations of the game
starting_visboard = np.vstack((np.ones((2,8))*ROBOT, np.zeros((4,8)), np.ones((2,8))*HUMAN))
pyboard = chess.Board()
cboard = defBoardCoords()

if ROBOT == chess.WHITE: # if robot is playing white, have it go first
    print("My turn first")
    robot_move, current_visboard = robotsVirtualMove(starting_visboard) # make the move virtually
    robotsPhysicalMove(robot_move) # make the move physically
else:
    current_visboard = starting_visboard
    print("Your turn first")
    stockboard = stockfish.get_board_visual(HUMAN == chess.WHITE)
    print(stockboard)

# take turns until the game ends. Each iteration starts when the human finishes their turn. Iteration includes processing the human's turn, then making a countermove.
while humansTurnFinished(): # eventually this will be based on the clock change, not commandline input
    
    ### HUMAN'S TURN ###
    # figure out what their move was
    seen_visboard, human_move = perceiveHumanMove()

    # if move is illegal make human try again
    if chess.Move.from_uci(human_move) not in pyboard.legal_moves:
        print("Not a valid move. Try again, human")
        if pyboard.is_into_check(chess.Move.from_uci(human_move)):
            print("PS, you might want to avoid check this time")
        
        # TODO: start human's timer again
        continue
    
    # update board representations
    pyboard.push_uci(human_move)
    stockfish.make_moves_from_current_position([human_move])
    
    # end the game if the human won   
    if gameOver():
        break
    
    ### ROBOT'S TURN ###
    # make the move virtually
    robot_move, current_visboard = robotsVirtualMove(seen_visboard)
    
    # make the move physically
    robotsPhysicalMove(robot_move)

    # end the game if the robot won
    if gameOver():
        break

    # TODO: currently the code doesn't handle castling (i don't even know what the algebraic notation is)
    # TODO: currently doesn't capture a piece before moving
    # TODO: make a better display of who won. The chess object output doesn't make sense
    # TODO: make waypoints that are more smooth and won't result in a jerky motion straight up, stop, over, stop, down.