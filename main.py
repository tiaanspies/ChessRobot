"""
### PSEUDOCODE ###
INITIALIZE
imports libraries, define functions

SETUP GAME
have human pick a side to play
start an instance of the stockfish engine
create representations of the game in numpy-chess, physical coordinates, and in -1, 0, 1 representation for the vision

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

import os
from stockfish import Stockfish
import numpy as np
import chess
import matplotlib.pyplot as plt
from Chessboard_detection import Fake_Camera, Chess_Vision
# from NLinkArm3d import NLinkArm -- commented out bc only necessary for IK and simulation

### INITIALIZE ###

# functions for gameplay
def initializeCamera():
    """Sets up the camera, needs to calibrate on both empty and starting board. returns cam and board instances"""
    CAMERA_RESOLUTION = (640, 480)

    # Open Video camera
    # cam = cv.VideoCapture(0)
    dirPath = os.path.dirname(os.path.realpath(__file__))
    relPath = "\\Chessboard_detection\\TestImages\\Set_2_W_Only"
    cam = Fake_Camera.FakeCamera(CAMERA_RESOLUTION, dirPath + relPath)    

    if not cam.isOpened():
        raise("Cannot open camera.")

    # Initialize ChessBoard object and select optimal thresh
    # Board must be empty when this is called
    ans = input("Is the empty board in view? (y/n): ").strip().lower()
    if ans == 'y':
        s, img = cam.read()
        board = Chess_Vision.ChessBoard(img)
    else:
        print("Error: put the board in view")
        exit()

    # NB --- Board is setup in starting setup.
    # Runs kmeans clustering to group peice and board colours
    ans = input("Are all the pieces placed now? (y/n): ").strip().lower()
    if ans== 'y':
        s, img = cam.read()
        board.fitKClusters(img)
    else:
        print("Error: set up the board")
        exit()

    return cam, board

def whichColor():
    """Human decides which color to play. Black is -1, White is 1"""
    ans = input("Hello, human. Are you playing black or white today? (b/w): ").strip().lower()
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
    """Simple yes/no question to determine if human is done with their turn. This will eventually be the function that flags when the clock is pressed"""

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

def gameOver():
    """returns False if game isn't over, prints the outcome and returns True if it is"""
    if pyboard.outcome():
        print("Good game, human")
        print(pyboard.outcome())     
        return True
    return False

# functions for handling visboard (the -1, 0, 1 representation)
def seeBoardReal():
    """Uses CV to determine which squares are occupied, returns -1, 0, 1 representation"""
    s, img = cam.read()  # read in image from camera
    positions = np.flip(board.getCurrentPositions(img),0) # turn it into -1, 0, 1 representation
    print(positions)
    return positions

def seeBoardFiller(board):
    """Filler function for seeBoardReal which takes users input for which piece to move and updates visboard with it"""
        
    ans = input("what is your move? ")
    board = updateVisBoard(board, ans, HUMAN)

    return board

def updateVisBoard(board, move, player, capture=None):
    """Updates the -1, 0, 1 representation of the board"""
    board_new = board.copy()
    start_square = chess.parse_square(move[:2])
    end_square = chess.parse_square(move[2:])

    temp = board_new.ravel()

    if capture:
        capture_square = chess.parse_square(capture)
        temp[capture_square] = 0
    
    temp[start_square] = 0
    temp[end_square] = player
    board_final = temp.reshape((8,8))
    
    return board_final
    
def compareVisBoards(current, previous):
    """compares the CV output with most recent board and outputs the move that was made or None if it can't tell"""
    
    # debugging
    print("current:")
    print(current)
    print("previous:")
    print(previous)
    print("compared:")
    print(current!=previous)
    print("Human:")
    print(current==HUMAN)
    

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
    
    human_move = start_name + end_name
    print(f"Percieved move was: {human_move}") # for debugging

    return human_move

def perceiveHumanMove(previous_visboard):
    """take image (or typed move for now) and return the move that was made"""
    # seen_visboard = seeBoardFiller(current_visboard.copy())   
    new_visboard = seeBoardReal() # Tiaan's vision function to find occupied squares
    human_move = compareVisBoards(new_visboard, previous_visboard) # Compare boards to figure out what piece moved
    if human_move is None:
        return perceiveHumanMove()
    return new_visboard, human_move

# functions for handling transition to real 3D space
def getLinePoints(start, goal, step):
    """Creates a 3xN nparray of 3D waypoints roughly 'step' distance apart between two 3D points"""
    dist = np.linalg.norm(goal - start)
    n_steps = int(dist // step)
    x_points = np.linspace(start[0], goal[0], n_steps)
    y_points = np.linspace(start[1], goal[1], n_steps)
    z_points = np.linspace(start[2], goal[2], n_steps)
    return np.vstack((x_points, y_points, z_points))

def defBoardCoords(square_width=30, border_width=10, base_dist=15, base_height=10):
    """gives an 3x8x8 ndarray representing the euclidean coordinates of the center of each board square"""
    # zero is at the robot base, which is centered base_dist from the edge of the board
    
    # initialize
    board_coords = np.zeros((3,8,8))
    board_width = 8 * square_width + 2 * border_width
    home = np.array([0, base_dist, 1.5*board_width])
    
    # set z coord
    board_coords[2,:,:] = base_height

    # define and set x and y coords of board
    file_coords = np.linspace(-3.5*square_width,3.5*square_width,8,endpoint=True)
    rank_coords = np.linspace(7.5*square_width,0.5*square_width,8,endpoint=True) + border_width + base_dist
    board_coords[:2,:,:] = np.array(np.meshgrid(file_coords,rank_coords))

    # define array for coords of captured pieces
    storage_coords = list(np.vstack((np.linspace(-board_width/2,board_width/2,15,endpoint=True),
                                np.ones(15)*(base_dist - square_width),
                                np.zeros(15))).T)

    return board_coords, storage_coords, home

def getCoords(square_name, board_coords):
    """gives real-world coordinates in mm based on chess board square (e.g. 'e2')"""
    file_idx = chess.FILE_NAMES.index(square_name[0]) if ROBOT==1 else chess.FILE_NAMES[::-1].index(square_name[0])
    rank_idx = chess.RANK_NAMES[::-1].index(square_name[1]) if ROBOT==1 else chess.RANK_NAMES.index(square_name[1])
    return board_coords[:,rank_idx,file_idx]

def getPath_simple(start, goal, capture_square, storage_list, lift=50, step=10):
    """creates a 3xN array of waypoints from the 3x1 start to the 3x1 end"""
    lift = np.array([0,0,lift])
    if capture_square is not None:
        storage = np.array(storage_list.pop(0))
        first_moves = np.hstack((getLinePoints(home, capture_square, step),
                                 capture_square.reshape((3,1)), capture_square.reshape((3,1)),
                                 getLinePoints(capture_square, capture_square + lift, step),
                                 getLinePoints(capture_square + lift, storage + lift, step),
                                 getLinePoints(storage + lift, storage, step),
                                 storage.reshape((3,1)), storage.reshape((3,1)),
                                 getLinePoints(storage, storage + lift, step),
                                 getLinePoints(storage + lift, start, step)))
    else:
        first_moves = getLinePoints(home, start, step)
    
    second_moves = np.hstack((start.reshape((3,1)), start.reshape((3,1)),
                              getLinePoints(start, start + lift, step),
                              getLinePoints(start + lift, goal + lift, step),
                              getLinePoints(goal + lift, goal, step),
                              goal.reshape((3,1)), goal.reshape((3,1)),
                              getLinePoints(goal, home, step)))

    path = np.hstack((first_moves, second_moves))
    
    # for debugging
    sq_width = 30
    bdr_width = 10
    bas_dist = 15
    file_lines = np.linspace(-4*sq_width,4*sq_width,9,endpoint=True)
    rank_lines = np.linspace(8*sq_width,0,9,endpoint=True) + bdr_width + bas_dist
    X,Y = np.meshgrid(file_lines,rank_lines)
    Z = np.ones_like(X) * 10
    ax = plt.axes(projection='3d')
    ax.plot_wireframe(X,Y,Z, color="r")
    ax.scatter3D(path[0],path[1],path[2])
    plt.show()
    
    return path

# functions for grouping sections of code
def robotsVirtualMove(visboard, human_move=None):
    """takes in the game in it's current state and returns it having made one best move or None if robot won"""
    
    # ask the engine for the best move -- instead right now, ask what move was made in the recorded game
    # best_move = stockfish.get_best_move()
    best_move = input("What was the robot's move? ")

    # handle captures
    capture = stockfish.will_move_be_a_capture(best_move)
    if capture is stockfish.Capture.DIRECT_CAPTURE:
        capture_square = best_move[2:]
    elif capture is stockfish.Capture.EN_PASSANT:
        capture_square = human_move[2:]
    elif capture is stockfish.Capture.NO_CAPTURE:
        capture_square = None

    # make that move and update the board representations
    stockfish.make_moves_from_current_position([best_move])
    visboard = updateVisBoard(visboard, best_move, ROBOT, capture=capture_square)
    pyboard.push_uci(best_move)

    # print the board so the human can make a next move - this won't be needed later
    stockboard = stockfish.get_board_visual(HUMAN == chess.WHITE)
    print(stockboard)

    return best_move, visboard, capture_square

def robotsPhysicalMove(robot_move, capture_square):
    """creates and executes the robot's physical move"""
    start = getCoords(robot_move[:2],cboard)
    goal = getCoords(robot_move[2:],cboard)
    if capture_square is not None:
        capture_square = getCoords(capture_square, cboard)
    path = getPath_simple(start, goal, capture_square, storage_list)
    
    # TODO: find thetapath using inverse kinematics
    
    return

# functions for simulation
def defRobotArm(L1=250,L2=250):
    """creates an instance of the NLinkArm class for our specific robot configuration"""
    # define robot parameters in DH convention [theta alpha a d]
    l1_params = [0, np.pi/2, 0, 0]
    l2_params = [0, 0, L1, 0]
    l3_params = [0, 0, L2, 0]
    param_list = [l1_params, l2_params, l3_params]
    return NLinkArm(param_list)

### Global variables ###
# create an instance of of the stockfish engine with the parameters requested
stockfish = Stockfish(r"C:\Users\HP\Documents\Chess Robot\stockfish\stockfish_15_win_x64_popcnt\stockfish_15_x64_popcnt.exe", depth=15, parameters={"UCI_Elo":800})

# create an instance of the cam and board classes for converting input from the camera
cam, board = initializeCamera()

# create variables for who is playing which side (1 = white, -1 = black)
HUMAN, ROBOT = whichColor()

# Define the -1, 0, 1 (visboard), python-chess (pyboard), and coordinate (cboard) representations of the game
starting_visboard = np.vstack((np.ones((2,8), dtype=np.int64)*ROBOT, np.zeros((4,8), dtype=np.int64), np.ones((2,8), dtype=np.int64)*HUMAN))
print(starting_visboard)
pyboard = chess.Board()
cboard, storage_list, home = defBoardCoords()

# create NLinkArm instance specific to our robot using python_robotics NLinkArm class -- commented out bc only necessary for IK and simulation
# robotarm = defRobotArm()

def main():

    if ROBOT == chess.WHITE: # if robot is playing white, have it go first
        print("My turn first")
        robot_move, current_visboard, capture_square = robotsVirtualMove(starting_visboard) # make the move virtually
        robotsPhysicalMove(robot_move, capture_square) # make the move physically
    else:
        current_visboard = starting_visboard
        print("Your turn first")
        stockboard = stockfish.get_board_visual(HUMAN == chess.WHITE)
        print(stockboard)

    # take turns until the game ends. Each iteration starts when the human finishes their turn. Iteration includes processing the human's turn, then making a countermove.
    while humansTurnFinished(): # eventually this will be based on the clock change, not commandline input
        
        ### HUMAN'S TURN ###
        # figure out what their move was
        seen_visboard, human_move = perceiveHumanMove(current_visboard)

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
        robot_move, current_visboard, capture_square = robotsVirtualMove(seen_visboard, human_move)
        
        # make the move physically
        robotsPhysicalMove(robot_move, capture_square)

        # end the game if the robot won
        if gameOver():
            break

    cam.release()

# TODO: currently the code doesn't handle castling (i don't even know what the algebraic notation is)
# TODO: make a better display of who won. The chess object output doesn't make sense
# TODO: make waypoints that are more smooth and won't result in a jerky motion straight up, stop, over, stop, down.

if __name__ == "__main__":
    main()