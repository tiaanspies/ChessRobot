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
import logging
import Chessboard_detection.pi_debugging as debug
### INITIALIZE ###

# define global variables for tracking the running score
HUMAN_SCORE = 0
ROBOT_SCORE = 0
GAME_COUNTER = 0

# functions for gameplay
def humansTurnFinished():
    """Simple yes/no question to determine if human is done with their turn. This will eventually be the function that flags when the clock is pressed"""
    
    ans = input("Are you finished? (y/n): ").strip().lower()
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
        global GAME_COUNTER
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
            parameters={"UCI_Elo":400, "UCI_LimitStrength":True},
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

    def compute_robot_move(self, human_move=None):
        """takes in the game in it's current state and returns it having made one best move or None if robot won"""
        # Move Cases
        # 1. Direct capture
        # 2. En passant
        # 3. Promotion
        # 4. Castling
        # 5. Regular move
        best_move = self.stockfish.get_best_move()

        # handle captures
        capture = self.stockfish.will_move_be_a_capture(best_move)
  
        if capture is self.stockfish.Capture.DIRECT_CAPTURE:
            capture_square = best_move[2:]
        elif capture is self.stockfish.Capture.EN_PASSANT:
            capture_square = human_move[2:]
            #TODO: Figure out what this human move is used for
        elif capture is self.stockfish.Capture.NO_CAPTURE:
            capture_square = None

        # make that move and update the board representations
        self.stockfish.make_moves_from_current_position([best_move])
        self.update_visboard(best_move, capture=capture_square)
        self.pyboard.push_uci(best_move)

        print(f"Robot's move: {best_move}")

        return best_move, capture_square
    
    def compute_human_move(self, human_move):
        # update board representations
        self.pyboard.push_uci(human_move)
        self.stockfish.make_moves_from_current_position([human_move])
    
    def update_visboard(self, move, capture=None):
        """Updates the -1, 0, 1 representation of the board"""
        board_new = self.current_visboard.copy()

        player = board_new.ravel()[chess.parse_square(move[:2])]
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
        
        # if move was a promotion, find out which piece they chose
        if chess.Move.from_uci(human_move + "q") in self.pyboard.legal_moves:
            human_move += input("Which piece did you promote the pawn to? [q,r,b,n]: ")
        
        return human_move
    
    def compare_visboards(self):
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
        if np.sum(self.current_visboard!=self.prev_visboard) >= 4:
            if self.human_color == chess.WHITE:
                if self.pyboard.has_kingside_castling_rights(chess.WHITE) and (np.sum(self.current_visboard.ravel()[4:7] != self.prev_visboard.ravel()[4:7]) == 3):
                    human_move = "e1g1"
                elif self.pyboard.has_queenside_castling_rights(chess.WHITE) and (np.sum(self.current_visboard.ravel()[2:5] != self.prev_visboard.ravel()[2:5]) == 3):
                    human_move = "e1c1"
                else:
                    ans = input("Detected 4+ changed pieces, but no one castled. Have you reset the turn and played again? ")
                    return None
            else: # Robot is white
                if self.pyboard.has_kingside_castling_rights(chess.BLACK) and (np.sum(self.current_visboard.ravel()[60:63] != self.prev_visboard.ravel()[60:63]) == 3):
                    human_move = "e8g8"
                elif self.pyboard.has_queenside_castling_rights(chess.BLACK) and (np.sum(self.current_visboard.ravel()[58:61] != self.prev_visboard.ravel()[58:61]) == 3):
                    human_move = "e8c8"
                else:
                    ans = input("Detected 4+ changed pieces, but no one castled. Have you reset the turn and played again? ")
                    return None
        
        # deal with any other type of move
        else:
            start_square = np.flatnonzero(np.logical_and((self.current_visboard!=self.prev_visboard),(self.current_visboard==0)))
            end_square = np.flatnonzero(np.logical_and((self.current_visboard!=self.prev_visboard),(self.current_visboard==self.human_color)))
            
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
    
    def is_legal_move(self, move):
        """checks if the move is legal"""
        # if move is illegal make human try again
        if chess.Move.from_uci(move) not in self.pyboard.legal_moves:
            print("Not a valid move. Try again, human")
            if self.pyboard.is_into_check(chess.Move.from_uci(move)):
                print("PS, you might want to avoid check this time")
            
            # TODO: replace the below with starting human's timer again
            ans = input("Have you reset the turn and played again? ")
            
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
        try:
            human_move = chess_manager.perceive_human_move(robot.cam)
        except Exception as e:
            print(e)
            print("Failed to perceive move. Trying again.")
            robot.move_home()
            human_move = chess_manager.perceive_human_move(robot.cam)
        
        # if not a legal move, have them try again
        if not chess_manager.is_legal_move(human_move):
            continue

        chess_manager.compute_human_move(human_move)
        
        # end the game if the human won   
        if chess_manager.is_game_over():
            break
        
        ### ROBOT'S TURN ###
        # make the move virtually
        robot_move, capture_square = chess_manager.compute_robot_move(human_move)
        
        # make the move physically
        robot.execute_chess_move(robot_move, capture_square)

        # end the game if the robot won
        if chess_manager.is_game_over():
            break

# TODO: make a better display of who won. The chess object output doesn't make sense - think I fixed this, will need to test
# TODO: handle promotions - done for human, not for robot
# TODO: handle castling for robot
# TODO: make waypoints that are more smooth and won't result in a jerky motion straight up, stop, over, stop, down. - started this with quintecs
# TODO: build in a check that the physical move worked according to the camera
# TODO: Add a way to reset the board if the human makes a mistake

if __name__ == "__main__":
    while GAME_COUNTER == 0 or play_again():
        main()