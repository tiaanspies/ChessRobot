# "\Test_Images\\b_w_game_2\\"
import Fake_Camera
import Chess_Vision

CAMERA_RESOLUTION = (640, 480)

def main():
    pass

def testFolder(folderPath):
    # cam = cv.VideoCapture(0)
    cam = Fake_Camera.FakeCamera(CAMERA_RESOLUTION, folderPath)    

    if not cam.isOpened():
        raise("Cannot open camera.")

    # Initialize ChessBoard object and select optimal thresh
    # Board must be empty when this is called.
    board = Chess_Vision.ChessBoard(cam)

    # Board is setup in starting setup.
    # Runs kmeans clustering to group peice and board colours
    board.fitKClusters()

    # display video of chessboard with corners
    while cv.waitKey(1) != ord('q'):        
        positions = board.getCurrentPositions()
        print(positions)

    cam.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()