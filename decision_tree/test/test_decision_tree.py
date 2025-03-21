import numpy as np
import unittest
from decision_tree.src.decision_tree import MyDecisionTreeClassifier
from data.get_data import get_classification_data

class TestDecisionTreeClassifier(unittest.TestCase):
    def setUp(self):
        self.X, self.y = get_classification_data(n_samples=50, n_classes=3, n_features=5)
        self.model = MyDecisionTreeClassifier()

    def test_training(self):
        self.model.fit(self.X, self.y)

    def test_structure(self):
        self.model.fit(self.X, self.y)
        self.assertTrue(len(self.model.tree) > 0, "The tree should not be empty after fitting.")

    def test_pure_node(self):
        X_pure = np.array([[1, 1], [1, 1], [1, 1]])
        y_pure = np.array([0, 0, 0])
        self.model.fit(X_pure, y_pure)
        y_pred = self.model.predict(X_pure)
        self.assertTrue(np.all(y_pred == 0), "Pure nodes should always predict the same class.")

    def test_accuracy(self):
        """Check if accuracy is above a reasonable threshold"""
        self.model.fit(self.X, self.y)
        y_pred = self.model.predict(self.X)
        accuracy = np.mean(y_pred == self.y)
        print(f"Accuracy: {accuracy}")
        self.assertGreaterEqual(accuracy, 0.8)

if __name__ == "__main__":
    unittest.main()