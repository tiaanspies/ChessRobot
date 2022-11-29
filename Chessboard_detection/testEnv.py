import Fake_Camera
import Chess_Vision
import os
import numpy as np
import pickle

CAMERA_RESOLUTION = (640, 480)

def main():
    # "\Test_Images\\b_w_game_2\\"
    folderNames = findFolders("TestImages")
    successRate = []
    for folder in folderNames:
        if "Test_Set_" in folder:
            # try:
            successRate.append(runTest(folder))
            # except:
            #     successRate.append(0)
    
    print("Tests Success rate: ")
    print(successRate)

def findFolders(relPath):
    """Find all folder in subfolder"""
    dirPath = os.path.dirname(os.path.realpath(__file__))
    fullTestPath = os.path.join(dirPath, relPath)

    testFolderNames = []
    for name in os.listdir(fullTestPath):
        fullPath = os.path.join(fullTestPath, name)
        if os.path.isdir(fullPath):
            testFolderNames.append(fullPath)
    
    return testFolderNames

def runTest(absFolderPath):
    cam = Fake_Camera.FakeCamera(CAMERA_RESOLUTION, absFolderPath)    

    if not cam.isOpened():
        raise("Cannot open camera.")

    # Initialize ChessBoard object and select optimal thresh
    # Board must be empty when this is called
    s, img = cam.read()
    board = Chess_Vision.ChessBoard(img)

    # NB --- Board is setup in starting setup.
    # Runs kmeans clustering to group peice and board colours
    s, img = cam.read()
    board.fitKClusters(img)

    # display video of chessboard with corners
    s, img = cam.read()
    
    # file = open(absFolderPath+"\\CorrectResults.pkl", "rb")
    with open(absFolderPath+"\\CorrectResults.pkl", "rb") as file:
        correctArr = pickle.load(file)
    # correctArr = []

    testsPassed = 0
    index = 0
    while s is True:  
        positions = board.getCurrentPositions(img)

        if np.equal(positions, correctArr[index]).all():
            testsPassed += 1
        # correctArr.append(positions)
        print(positions)
        index += 1

        s, img = cam.read()  
        
    # # # # Used to save test files
    # correctArr = np.array(correctArr)
    # with open(absFolderPath+"\\CorrectResults.pkl", "wb") as file:    
    #     pickle.dump(correctArr, file)

    cam.release()
    return testsPassed/index

if __name__ == "__main__":
    main()