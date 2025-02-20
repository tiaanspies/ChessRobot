from Chessboard_detection.test_manager import TestChessLabeling

if __name__ == "__main__":
    tester = TestChessLabeling()
    tester.setUpClass()
    tester.setUp()
    tester.test_labeling_accuracy()