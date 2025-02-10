import json
import unittest
import numpy as np
from collections import defaultdict
from pathlib import Path
import cv2
from Chessboard_detection.projection import label_chessboard, label_dict_to_str

DATASET_PATH = Path("Chessboard_detection", "dataset")

class TestChessLabeling(unittest.TestCase):
    """Tests for chessboard square labeling"""

    @classmethod
    def setUpClass(cls):
        """Runs once before all tests"""
        print("\nSetting up test environment...")
        cls.image_dir = DATASET_PATH / "images"
        cls.label_dir = DATASET_PATH / "labels"
        cls.test_images = [f.name for f in cls.image_dir.iterdir() if f.suffix == ".jpg"]
        print(f"Found {len(cls.test_images)} test images")
        for img in cls.test_images:
            print(f" - {img}")
    def setUp(self):
        """Runs before each test case"""
        self.confusion_matrix = defaultdict(lambda: defaultdict(int))
        self.total = 0
        self.correct = 0

    def test_labeling_accuracy(self):
        """Evaluate labeling accuracy, breaking down results by misclassification types"""
        print("\nEvaluating labeling accuracy...")
        for img_name in self.test_images:
            image_path = self.image_dir / img_name
            label_path = self.label_dir / img_name.replace(".jpg", ".json")

            with open(str(label_path), "r") as f:
                expected_labels = json.load(f)

            image = cv2.imread(str(image_path))
            predicted_label_dict = label_chessboard(image)
            predicted_labels = label_dict_to_str(predicted_label_dict)

            for square, expected in expected_labels.items():
                expected_label = f"{expected['piece']} on {expected['background']}"
                predicted_label = predicted_labels.get(square, "missing")

                # Update confusion matrix
                self.confusion_matrix[expected_label][predicted_label] += 1

                # Accuracy count
                if expected_label == predicted_label:
                    self.correct += 1
                self.total += 1

        accuracy = self.correct / self.total if self.total > 0 else 0
        self.display_results(accuracy)

        self.assertEqual(accuracy, 1, "Accuracy below 100% threshold")

    def display_results(self, accuracy):
        """Print confusion matrix and accuracy"""
        print("\nConfusion Matrix:")
        categories = sorted(set(self.confusion_matrix.keys()) | 
                            {key for row in self.confusion_matrix.values() for key in row})
        
        # Print Header
        header = "{:<20}".format("Expect \\ Predict")
        for cat in categories:
            header += "{:<20}".format(cat)
        print(header)

        # Print Rows
        for exp in categories:
            row = "{:<20}".format(exp)
            for pred in categories:
                row += "{:<20}".format(self.confusion_matrix[exp][pred])
            print(row)

        print(f"\nOverall Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    unittest.main()