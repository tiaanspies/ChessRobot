from stockfish import Stockfish

## SETUP
# give path to where the stockfish engine .exe file is stored locally
localpath = None # can't figure out how to make the path into a variable

# Change the default settings as needed
depth = 15
params = {
    "Debug Log File": "",
    "Contempt": 0,
    "Min Split Depth": 0,
    "Threads": 1, # More threads will make the engine stronger, but should be kept at less than the number of logical processors on your computer.
    "Ponder": "false",
    "Hash": 16, # Default size is 16 MB. It's recommended that you increase this value, but keep it as some power of 2. E.g., if you're fine using 2 GB of RAM, set Hash to 2048 (11th power of 2).
    "MultiPV": 1,
    "Skill Level": 20,
    "Move Overhead": 10,
    "Minimum Thinking Time": 20,
    "Slow Mover": 100,
    "UCI_Chess960": "false",
    "UCI_LimitStrength": "false",
    "UCI_Elo": 1350
}

# initialize the class using the path and settings
stockfish = Stockfish(r"C:\Users\HP\Documents\Chess Robot\stockfish\stockfish_15_win_x64_popcnt\stockfish_15_x64_popcnt.exe", depth=depth, parameters=params)


'''
#### PSEUDOCODE ####
SETUP
move the arm to starting location
something to initialize whether robot is black or white

GAMEPLAY
when it's the robots turn (based on clock change):
    - get occupied squares (Tiaan's CV function)
    - compare to previous and determine what the human move was (output string, ie "e2e4")
    - make the robot's move
        -- update memory with human's move (stockfish.make_moves_from_current_position())
        -- get the best move (stockfish.get_best_move())
        -- solve path planner for how to execute that move
            --- remove captured piece
            --- move current piece
        -- execute the move and return to starting location
        -- update memory with robot's move (stockfish.make_moves_from_current_position())
    - change the clock and wait for next turn

'''
stockfish.make_moves_from_current_position
