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

### INITIALIZE ###

# necessary functions
def whichColor():
    """Human decides which color to play"""
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
    
    ans = input("what was your move? ")
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

    return visboard

def perceiveHumanMove():
    """take image (or typed move for now) and return the move that was made"""
    seen_visboard = seeBoard(current_visboard.copy()) # Tiaan's vision function to find occupied squares  
    human_move = compareVisBoards(seen_visboard, current_visboard) # Compare boards to figure out what piece moved
    if human_move is None:
        return perceiveHumanMove()
    return seen_visboard, human_move

# initialize stockfish
depth = 15
params = {
    "Contempt": 0,
    "Threads": 1, # More threads will make the engine stronger, but should be kept at less than the number of logical processors on your computer.
    "Hash": 128, # Default size is 16 MB. It's recommended that you increase this value, but keep it as some power of 2. E.g., if you're fine using 2 GB of RAM, set Hash to 2048 (11th power of 2).
    "UCI_LimitStrength": "true",
    "UCI_Elo": 800
}
stockfish = Stockfish(r"C:\Users\HP\Documents\Chess Robot\stockfish\stockfish_15_win_x64_popcnt\stockfish_15_x64_popcnt.exe", depth=depth, parameters=params)

### SETUP GAME ###

# have human pick which side to play (1 = white, -1 = black)
HUMAN, ROBOT = whichColor()

# Define the -1, 0, 1 representation (visboard) and python-chess (pyboard) versions of the game
starting_visboard = np.vstack((np.ones((2,8))*ROBOT, np.zeros((4,8)), np.ones((2,8))*HUMAN))
pyboard = chess.Board()

if ROBOT == chess.WHITE: # if robot is playing white, have it go first
    print("My turn first")
    current_visboard = robotsVirtualMove(starting_visboard) # make the move virtually
    # TODO: implement the physical commands to the robot # make the move physically
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
    current_visboard = robotsVirtualMove(seen_visboard)
    
    # make the move physically
    # TODO: implement the physical commands to the robot

    # end the game if the robot won
    if gameOver():
        break