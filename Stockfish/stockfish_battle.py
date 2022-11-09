from stockfish import Stockfish

## SETUP

# Change the default settings as needed
white_depth = 2
white_params = {
    "Debug Log File": "",
    "Contempt": 0,
    "Min Split Depth": 0,
    "Threads": 1, # More threads will make the engine stronger, but should be kept at less than the number of logical processors on your computer.
    "Ponder": "false",
    "Hash": 16, # Default size is 16 MB. It's recommended that you increase this value, but keep it as some power of 2. E.g., if you're fine using 2 GB of RAM, set Hash to 2048 (11th power of 2).
    "MultiPV": 1,
    "Skill Level": 10,
    "Move Overhead": 10,
    "Minimum Thinking Time": 20,
    "Slow Mover": 100,
    "UCI_Chess960": "false",
    "UCI_LimitStrength": "false",
    "UCI_Elo": 1350
}
black_depth = 2
black_params = {
    "Debug Log File": "",
    "Contempt": 0,
    "Min Split Depth": 0,
    "Threads": 1, # More threads will make the engine stronger, but should be kept at less than the number of logical processors on your computer.
    "Ponder": "false",
    "Hash": 16, # Default size is 16 MB. It's recommended that you increase this value, but keep it as some power of 2. E.g., if you're fine using 2 GB of RAM, set Hash to 2048 (11th power of 2).
    "MultiPV": 1,
    "Skill Level": 1,
    "Move Overhead": 10,
    "Minimum Thinking Time": 20,
    "Slow Mover": 100,
    "UCI_Chess960": "false",
    "UCI_LimitStrength": "false",
    "UCI_Elo": 1350
}

# initialize the class using the path and settings
white_stockfish = Stockfish(r"C:\Users\HP\Documents\Chess Robot\stockfish\stockfish_15_win_x64_popcnt\stockfish_15_x64_popcnt.exe", depth=white_depth, parameters=white_params)
black_stockfish = Stockfish(r"C:\Users\HP\Documents\Chess Robot\stockfish\stockfish_15_win_x64_popcnt\stockfish_15_x64_popcnt.exe", depth=black_depth, parameters=black_params)

def print_board(game, who_played):
    """ prints a simple representation of the game and the last two moves """
    
    # print the most recent moves
    print(who_played + str(game[-1]))
    
    # tell the engine about the game
    black_stockfish.set_position(game)

    # display the board
    print(black_stockfish.get_board_visual(), end='\n')
    return

def make_best_move(game, player):
    """ takes in the game in it's current state and returns it having made one best move """
    
    # tell the engine about the game and ask for the best move
    player.set_position(game)
    best_move = player.get_best_move()

    # flag checkmate/stalemate
    if not best_move:
        return False

    # make that move and update our list of positions
    player.make_moves_from_current_position([best_move])
    game.append(best_move)
    return game

def main():
    # set total number of moves to play
    MOVE_MAX = 200
    
    # starting board
    game = []
        
    # white's first move
    move_count = 0
    first_move = "e2e4"
    white_stockfish.make_moves_from_current_position(["e2e4"])
    game.append(first_move) 
    
    # play move_max more moves
    while move_count < MOVE_MAX and game:
        
        move_count += 1
        
        if move_count % 2: # black plays
            print_board(game, "White played ")
            game = make_best_move(game, white_stockfish)

        else: # white plays
            print_board(game, "Black played ")
            game = make_best_move(game, black_stockfish)
    
    # Print the results
    print("number of moves = " + str(move_count))
    if not game:
        print(["White won" if move_count % 2 else "Black won"])
    else:
        print("Not enough moves.")

main()