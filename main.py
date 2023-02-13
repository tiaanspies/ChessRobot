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
from IK_Solvers.traditional import ChessMoves
from motor_commands import MotorCommands
import picamera

### INITIALIZE ###

# define global variables for tracking the running score
HUMAN_SCORE = 0
ROBOT_SCORE = 0
GAME_COUNTER = 0

# functions for gameplay
def initializeCamera():
    """Sets up the camera, needs to calibrate on both empty and starting board. returns cam and board instances"""
    CAMERA_RESOLUTION = (640, 480)

    # Open Video camera
    # cam = cv.VideoCapture(0)
    dirPath = os.path.dirname(os.path.realpath(__file__))
    relPath = "\\Chessboard_detection\\TestImages\\Set_2_W_Only"
    cam = Fake_Camera.FakeCamera(CAMERA_RESOLUTION, dirPath + relPath) # Change .FakeCamera to .PhoneCamera
    cam = PiCamera()
    if not cam.isOpened():
        raise("Cannot open camera.")

    # Initialize ChessBoard object and select optimal thresh
    # Board must be empty when this is called
    ans = input("Is the empty board in view? (y/n): ").strip().lower()
    if ans == 'y':
        s, img = cam.read()
        board = Chess_Vision.ChessBoard(img)
    else:
        print("Please put the empty board is in view")
        initializeCamera()

    return cam, board

def identifyColors():
    """Runs k-means to set color centroids, then uses this to determine who's playing which color"""
    # NB --- Board is setup in starting setup.
    # Runs kmeans clustering to group piece and board colours
    ans = input("Are all the pieces placed now? (y/n): ").strip().lower()
    if ans== 'y':
        s, img = cam.read()
        HUMAN, ROBOT = board.initBoardWithStartPos(img)
    else:
        print("Please set up the board")
        identifyColors()
    return HUMAN, ROBOT

def whichColor():
    """OBSOLETE. Human decides which color to play. Black is -1, White is 1"""
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
    # return True
    
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
    outcome = pyboard.outcome()
    
    if outcome is not None:

        # print outcome type
        print(outcome.termination.name)

        # if there was a winner update the running scores
        if outcome.winner is not None:
            if outcome.winner == (HUMAN==chess.WHITE):
                global HUMAN_SCORE
                HUMAN_SCORE += 1
            else:
                global HUMAN_SCORE
                ROBOT_SCORE += 1
        
        # print the running scores
        print(f"Robot: {ROBOT_SCORE}, Human: {HUMAN_SCORE}")
        print("Good game, human")
        return True

    return False

def play_again():
    """decide whether to play again"""
    ans = input("Play again? (y/n): ").strip().lower()
    if ans == 'y':
        global GAME_COUNTER
        GAME_COUNTER += 1
        return True
    elif ans == 'n':
        print("That's okay, thanks for playing!")
        return False
    else:
        play_again()
        
# functions for handling visboard (the -1, 0, 1 representation)
def seeBoardReal():
    """Uses CV to determine which squares are occupied, returns -1, 0, 1 representation"""
    s, img = cam.read()  # read in image from camera
    positions = board.getCurrentPositions(img) # turn it into -1, 0, 1 representation
    visboard = np.fliplr(positions)
    # print(visboard)
    return visboard

def seeBoardFiller(board):
    """Filler function for seeBoardReal which takes users input for which piece to move and updates visboard with it"""
        
    ans = input("what is your move? ")
    board = updateVisBoard(board, ans, HUMAN)

    return board

def updateVisBoard(board, move, player, capture=None):
    """Updates the -1, 0, 1 representation of the board"""
    board_new = board.copy()

    # deal with castling
    if move == "e1g1":
        board_new.ravel()[4:8] = 0
        board_new.ravel()[5:7] = player
    elif move == "e8g8":
        board_new.ravel()[60:] = 0
        board_new.ravel()[61:63] = player       
    elif move == "e1c1":
        board_new.ravel()[0:5] = 0
        board_new.ravel()[2:4] = player
    elif move == "e8c8":
        board_new.ravel()[56:61] = 0
        board_new.ravel()[58:60] = player

    # deal with other moves
    else:
        start_square = chess.parse_square(move[:2])
        end_square = chess.parse_square(move[2:])

        # adding this here handles en passants
        if capture:
            capture_square = chess.parse_square(capture)
            board_new.ravel()[capture_square] = 0
        
        board_new.ravel()[start_square] = 0
        board_new.ravel()[end_square] = player
        
    return board_new
    
def compareVisBoards(current, previous):
    """compares the CV output with most recent board and outputs the move that was made or None if it can't tell"""
    
    # debugging
    '''
    print("current:")
    print(current)
    print("previous:")
    print(previous)
    print("compared:")
    print(current!=previous)
    print("Human:")
    print(current==HUMAN)
    '''

    # deal with castling
    if np.sum(current!=previous) >= 4:
        if HUMAN == chess.WHITE:
            if pyboard.has_kingside_castling_rights(chess.WHITE) and (np.sum(current.ravel()[4:7] != previous.ravel()[4:7]) == 3):
                human_move = "e1g1"
            elif pyboard.has_queenside_castling_rights(chess.WHITE) and (np.sum(current.ravel()[2:5] != previous.ravel()[2:5]) == 3):
                human_move = "e1c1"
            else:
                ans = input("Detected 4+ changed pieces, but no one castled. Have you reset the turn and played again? ")
                return None
        else: # Robot is white
            if pyboard.has_kingside_castling_rights(chess.BLACK) and (np.sum(current.ravel()[60:63] != previous.ravel()[60:63]) == 3):
                human_move = "e8g8"
            elif pyboard.has_queenside_castling_rights(chess.BLACK) and (np.sum(current.ravel()[58:61] != previous.ravel()[58:61]) == 3):
                human_move = "e8c8"
            else:
                ans = input("Detected 4+ changed pieces, but no one castled. Have you reset the turn and played again? ")
                return None
    
    # deal with any other type of move
    else:
        start_square = np.flatnonzero(np.logical_and((current!=previous),(current==0)))
        end_square = np.flatnonzero(np.logical_and((current!=previous),(current==HUMAN)))
        
        if start_square.size == 0:
            print("failed to locate which piece was moved")
            ans = input("Have you reset the turn and played again? ")
            return None
        elif end_square.size == 0:
            print("failed to locate where the piece was played")
            ans = input("Have you reset the turn and played again? ")
            return None
        
        start_name = chess.square_name(start_square[0])
        end_name = chess.square_name(end_square[0])
        
        human_move = start_name + end_name

    print(f"Percieved move was: {human_move}") # for debugging
    return human_move

def perceiveHumanMove(previous_visboard):
    """take image and return the move that was made"""
    # seen_visboard = seeBoardFiller(current_visboard.copy())   
    new_visboard = seeBoardReal() # Tiaan's vision function to find occupied squares
    human_move = compareVisBoards(new_visboard, previous_visboard) # Compare boards to figure out what piece moved
    
    # if move was not successfully detected, start over
    if human_move is None:
        return perceiveHumanMove(previous_visboard)
    
    # if move was a promotion, find out which piece they chose
    if chess.Move.from_uci(human_move + "q") in board.legal_moves:
        human_move += input("Which piece did you promote the pawn to? [q,r,b,n]: ")
    
    return new_visboard, human_move

# functions for handling transition to real 3D space
'''
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
    ax.view_init(90,270)
    plt.show(block=False)

    return path
'''

# functions for grouping sections of code
def robotsVirtualMove(visboard, human_move=None):
    """takes in the game in it's current state and returns it having made one best move or None if robot won"""
    
    # ask the engine for the best move -- instead right now, ask what move was made in the recorded game
    best_move = stockfish.get_best_move()
    # best_move = input("What was the robot's move? ")

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
    # stockboard = stockfish.get_board_visual(HUMAN == chess.WHITE)
    # print(stockboard)

    print(f"Robot's move: {best_move}")

    return best_move, visboard, capture_square

def robotsPhysicalMove(robot_move, capture_square):
    """creates and executes the robot's physical move"""
    start = cm.get_coords(robot_move[:2])
    goal = cm.get_coords(robot_move[2:])
    if capture_square is not None:
        capture_square = cm.getCoords(capture_square)
    path = cm.generate_quintic_path(start, goal, capture_square) # generate waypoints
    thetas = cm.inverse_kinematics(path) # convert to joint angles
    thetas = mc.add_gripper_commands(thetas) # remove unnecessary wrist commands, add gripper open close instead
    thetas = mc.fit_robot_limits(thetas)
    mc.run(thetas) # pass joint angles to motors
    
    # simulate
    # cm.plot_robot(thetas, path)

### Global variables ###
# create an instance of of the stockfish engine with the parameters requested
stockfish = Stockfish(r"C:\Users\HP\Documents\Chess Robot\stockfish\stockfish_15_win_x64_popcnt\stockfish_15_x64_popcnt.exe", depth=15, parameters={"UCI_Elo":500})

# create an instance of the cam and board classes for converting input from the camera
cam, board = initializeCamera()

# create an instance of the ChessMoves class, which holds all functions for converting a algebraic notation move to a theta trajectory
cm = ChessMoves() # this class takes all the board and robot measurements as optional args

# create an instance of the MotorCommands class, which is used to communicate with the raspberry pi
mc = MotorCommands()

# Define the -1, 0, 1 (visboard), python-chess (pyboard), and coordinate (cboard) representations of the game
starting_visboard = np.vstack((np.ones((2,8), dtype=np.int64), np.zeros((4,8), dtype=np.int64), np.ones((2,8), dtype=np.int64)*-1))
pyboard = chess.Board()
# cboard, storage_list, home = defBoardCoords()

def main():
    
    # determine who is playing which side and run k-means to set color centroids
    global HUMAN, ROBOT 
    HUMAN, ROBOT = identifyColors()

    if ROBOT == chess.WHITE: # if robot is playing white, have it go first
        print("My turn first")
        robot_move, current_visboard, capture_square = robotsVirtualMove(starting_visboard) # make the move virtually
        robotsPhysicalMove(robot_move, capture_square) # make the move physically
    else:
        current_visboard = starting_visboard
        print("Your turn first")
        
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
            
            # TODO: replace the below with starting human's timer again
            ans = input("Have you reset the turn and played again? ")
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

# TODO: make a better display of who won. The chess object output doesn't make sense - think I fixed this, will need to test
# TODO: handle promotions - done for human, not for robot
# TODO: handle castling for robot
# TODO: make waypoints that are more smooth and won't result in a jerky motion straight up, stop, over, stop, down. - started this with quintecs
# TODO: build in a check that the physical move worked according to the camera

if __name__ == "__main__":
    while GAME_COUNTER == 0 or play_again():
        main()