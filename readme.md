# Robot arm chess robot system

## Description
Chess robot will crush all opponents.

The project is currently in development. The goal for the system is have the robot placed next to any chessboard and have it interactively play against a human opponent. There are systems available like this but they have many limitations. This system will be compatable with many chessboards and pieces and be robust against various lighting conditions.

### Functional components:
-Robot vision
-Stockfish 15 Integration (State of the art chess engine)
-Robot arm path planning

### Installation:
pip install numpy
pip install matplotlib
pip install stockfish (also need to actually download something, I haven't figured that part out yet)
pip install chess
pip install requests
pip install opencv-python
pip install scipy
pip install scikit-learn