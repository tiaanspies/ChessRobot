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

from stockfish import Stockfish
import numpy as np
import chess
# from Chessboard_detection import Chess_Vision_kmeans
from Chessboard_detection.chess_vision_hough import ChessVisionHough
from Positioning.robot_manager import Robot
from time import sleep
### INITIALIZE ###

# define global variables for tracking the running score
HUMAN_SCORE = 0
ROBOT_SCORE = 0
GAME_COUNTER = 0

# functions for gameplay
def humansTurnFinished():
    """Simple yes/no question to determine if human is done with their turn. This will eventually be the function that flags when the clock is pressed"""
    
    ans = input("Your turn. Are you finished? (y/n): ").strip().lower()
    if ans not in ['y', 'n', '']:
        print(f'"{ans}" is invalid, please try again...')
        return humansTurnFinished()
    elif ans == 'n':
        print('Take your time, human')
        return humansTurnFinished()
    elif ans == 'y' or ans == '':
        return True
    return False

def play_again():
    """decide whether to play again"""
    ans = input("Play again? (y/n): ").strip().lower()
    if ans == 'y':
        GAME_COUNTER += 1
        return True
    elif ans == 'n':
        print("That's okay, thanks for playing!")
        return False
    else:
        play_again()

class ChessManager:
    def __init__(self):
        self.stockfish = Stockfish(
            r"/home/tpie/ChessRobot/Stockfish/Stockfish-sf_15/src/stockfish", 
            parameters={"UCI_Elo":400, "UCI_LimitStrength":'true'},
            depth=10
        )
        self.pyboard = chess.Board()
        self.human_color = None
        self.robot_color = None
        self.current_visboard = np.vstack((np.ones((2,8), dtype=np.int64), np.zeros((4,8), dtype=np.int64), np.ones((2,8), dtype=np.int64)*-1))
        self.prev_visboard = self.current_visboard.copy()
        self.board_vision = ChessVisionHough("black_white")

    def setup_board_vision_starting_position(self, cam):

        # Initialize ChessBoard object and select optimal thresh
        # Board must be empty when this is called
        while True:
            ans = input("Is the initial starting board setup in view? (y/n): ").strip().lower()
            if ans == 'y':
                _, img = cam.read()

                self.board_vision.setup_starting_position_board(img)

                return
            else:
                print("Please put the empty board is in view.")

    def identify_colors(self):
        """Runs k-means to set color centroids, then uses this to determine who's playing which color"""
        
        # Assume the human is white for now

        self.human_color = chess.WHITE
        self.robot_color = chess.BLACK

    def compute_robot_move(self):
        """takes in the game in it's current state and returns it having made one best move or None if robot won"""
        
        best_move = self.stockfish.get_best_move()

        return best_move
    
    def compute_human_move(self, human_move):
        # update board representations
        self.pyboard.push_uci(human_move)
        self.stockfish.make_moves_from_current_position([human_move])
    
    def update_visboard(self):
        """Updates the -1, 0, 1 representation of the board"""
        piece_map = self.pyboard.piece_map()
        board_new = np.zeros((8,8), dtype=np.int64)

        for square, piece in piece_map.items():
            row, col = divmod(square, 8)
            if piece.color == chess.WHITE:
                board_new[row, col] = 1
            else:
                board_new[row, col] = -1
            
        self.current_visboard = board_new

    def perceive_human_move(self, cam):
        """take image and return the move that was made"""
        # save current visboard into prev
        self.prev_visboard = self.current_visboard.copy()
        
        # get image
        _, img = cam.read()
        
        positions = self.board_vision.indentify_piece_ids(img) # turn it into -1, 0, 1 representation
        # self.current_visboard = np.fliplr(np.array(positions).reshape(8,8))
        self.current_visboard = np.array(positions).reshape(8,8)
        human_move = self.compare_visboards() # Compare boards to figure out what piece moved
        
        if human_move is None:
            self.current_visboard = self.prev_visboard.copy()
            return None
        
        # if move was a promotion, find out which piece they chose
        if chess.Move.from_uci(human_move + "q") in self.pyboard.legal_moves:
            human_move += input("Which piece did you promote the pawn to? [q,r,b,n]: ")
        
        return human_move
    
    def compare_visboards(self):
        """compares the CV output with most recent board and outputs the move that was made or None if it can't tell"""
        
        # deal with castling
        if np.sum(self.current_visboard!=self.prev_visboard) >= 4:
            if self.human_color == chess.WHITE:
                if self.pyboard.has_kingside_castling_rights(chess.WHITE) and (np.sum(self.current_visboard.ravel()[4:7] != self.prev_visboard.ravel()[4:7]) == 3):
                    human_move = "e1g1"
                elif self.pyboard.has_queenside_castling_rights(chess.WHITE) and (np.sum(self.current_visboard.ravel()[2:5] != self.prev_visboard.ravel()[2:5]) == 3):
                    human_move = "e1c1"
                else:
                    print("Detected 4+ changed pieces, but no one castled. Please reset and play again.")
                    return None
            else: # Robot is white
                if self.pyboard.has_kingside_castling_rights(chess.BLACK) and (np.sum(self.current_visboard.ravel()[60:63] != self.prev_visboard.ravel()[60:63]) == 3):
                    human_move = "e8g8"
                elif self.pyboard.has_queenside_castling_rights(chess.BLACK) and (np.sum(self.current_visboard.ravel()[58:61] != self.prev_visboard.ravel()[58:61]) == 3):
                    human_move = "e8c8"
                else:
                    print("Detected 4+ changed pieces, but no one castled. Please reset and play again.")
                    return None
        
        # deal with any other type of move
        else:
            start_square = np.flatnonzero(np.logical_and((self.current_visboard!=self.prev_visboard),(self.current_visboard==0)))
            end_square = np.flatnonzero(np.logical_and((self.current_visboard!=self.prev_visboard),(self.current_visboard==self.human_color)))
            
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
    
    def is_legal_move(self, move):
        """checks if the move is legal"""
        # if move is illegal make human try again
        if chess.Move.from_uci(move) not in self.pyboard.legal_moves:
            print("Not a valid move. I will be undoing that, human")
            if self.pyboard.is_into_check(chess.Move.from_uci(move)):
                print("PS, you might want to avoid check this time")
            
            return False
        return True
            
    def is_game_over(self):
        """returns False if game isn't over, prints the outcome and returns True if it is"""
        outcome = self.pyboard.outcome()
        
        if outcome is not None:

            # print outcome type
            print(outcome.termination.name)

            # if there was a winner update the running scores
            if outcome.winner is not None:
                if outcome.winner == (self.human_color==chess.WHITE):
                    global HUMAN_SCORE
                    HUMAN_SCORE += 1
                else:
                    global ROBOT_SCORE
                    ROBOT_SCORE += 1
            
            # print the running scores
            print(f"Robot: {ROBOT_SCORE}, Human: {HUMAN_SCORE}")
            print("Good game, human")
            return True

        return False
    
    def complete_robot_move(self, robot_move):
        # Move Cases
        # 1. Direct capture (no en passant)
        # 2. En passant 
        # 3. Promotion
        # 4. Castling
        # 5. Regular move

        promotion_request = None
        rook_move = None
        capture_square = None

        uci_move = chess.Move.from_uci(robot_move)
        # determine the type of capture
        if self.pyboard.is_en_passant(uci_move): # en passant
            start_rank = robot_move[1]
            end_file = robot_move[2]
            capture_square = end_file + start_rank

        elif robot_move[-1].lower() in ['q','r','b','n']: # promotion
            promotion_request = robot_move[-1].lower()
            if self.pyboard.is_capture(uci_move):
                capture_square = robot_move[2:4]

        elif self.pyboard.is_capture(uci_move): # direct capture
            capture_square = robot_move[2:]

        elif self.pyboard.is_castling(uci_move): # castling
            rank = robot_move[1]
            end_file = robot_move[2]
            if end_file == 'g':
                rook_move = 'h' + rank + 'f' + rank
            elif end_file == 'c':
                rook_move = 'a' + rank + 'd' + rank

        else: # regular move
            pass

        # make that move and update the board representations
        self.stockfish.make_moves_from_current_position([robot_move])
        self.pyboard.push_uci(robot_move)
        self.update_visboard()

        return capture_square, rook_move, promotion_request
        
def perceive_w_retries(chess_manager: ChessManager, robot: Robot):
    try:
        human_move = chess_manager.perceive_human_move(robot.cam)
    except:
        print("Whoops, I couldn't see the board. Trying again...")
        robot.move_home_backward()
        sleep(0.5)
        try:
            human_move = chess_manager.perceive_human_move(robot.cam)
        except:
            print("I still can't see. Tell Tiaan to improve his algorithm.")
            print("But just so we can keep playing. What move did you make? e.g.'b2b4'")
            human_move = input()

    return human_move

def main():
    #define managers

    robot = Robot()
    chess_manager = ChessManager()
    
    # move robot to starting position
    robot.move_home()
    robot.motor_commands.gripper.open_gripper()

    # create an instance of the cam and board classes for converting input from the camera
    chess_manager.setup_board_vision_starting_position(robot.cam)
    
    #determine which color the Robot and human are playing.
    chess_manager.identify_colors()

    if chess_manager.robot_color == chess.WHITE: # if robot is playing white, have it go first
        print("Sweet! Since I am white, my turn first")
        robot_move, capture_square = chess_manager.compute_robot_move() # make the move virtually
        robot.execute_chess_move(robot_move, capture_square) # make the move physically
    else:
        print("Your turn first")
        
    # take turns until the game ends. Each iteration starts when the human finishes their turn. Iteration includes processing the human's turn, then making a countermove.
    while humansTurnFinished(): # eventually this will be based on the clock change, not commandline input
        
        ### HUMAN'S TURN ###
        # figure out what their move was
        move_success = False

        while not move_success:
            print("===========================================")
            human_move = perceive_w_retries(chess_manager, robot)

            if human_move is None:
                humansTurnFinished()
                continue

            # if not a legal move, have them try again
            if not chess_manager.is_legal_move(human_move):
                move_reversed = human_move[2:4] + human_move[0:2]
                robot.execute_chess_move(move_reversed, None, None)
                print("That is not a legal move. Please try again.")
                chess_manager.update_visboard()
                sleep(1)
                humansTurnFinished()
                continue # undo move
            
            move_success = True
            chess_manager.compute_human_move(human_move)
        
        # end the game if the human won   
        if chess_manager.is_game_over():
            break
        
        ### ROBOT'S TURN ###
        # make the move virtually
        robot_move = chess_manager.compute_robot_move()
        
        # make the move physically
        capture_square, rook_move, promotion_request = chess_manager.complete_robot_move(robot_move)
        robot.execute_chess_move(robot_move, capture_square, rook_move)

        if promotion_request is not None:
            promote_dict = {'q': 'queen', 'r': 'rook', 'b': 'bishop', 'n': 'knight'}
            print(f"Please promote the pawn to a {promote_dict[promotion_request]}")
            input("Press enter when ready")

        # end the game if the robot won
        if chess_manager.is_game_over():
            break

# TODO: build in a check that the physical move worked according to the camera
# TODO: Add a way to reset the board if the human makes a mistake

if __name__ == "__main__":
    while GAME_COUNTER == 0 or play_again():
        GAME_COUNTER += 1
        main()