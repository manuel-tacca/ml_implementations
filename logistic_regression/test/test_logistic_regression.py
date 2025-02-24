import numpy as np
import unittest
from logistic_regression.src.logistic_regression import LogisticRegressionGD
from data.get_data import get_classification_data

class TestLogisticRegression(unittest.TestCase):
    def setUp(self):
        self.X, self.y = get_classification_data(n_samples=200, n_features=2)
        self.model = LogisticRegressionGD(learning_rate=0.1, n_iterations=1000)

    def test_training(self):
        self.model.fit(self.X, self.y)
        self.assertEqual(self.model.w.shape, (self.X.shape[1], 1))
        self.assertIsInstance(self.model.b, float)

    def test_prediction_accuracy(self):
        self.model.fit(self.X, self.y)
        y_pred = self.model.predict(self.X)
        accuracy = np.mean(y_pred == self.y)
        self.assertGreaterEqual(accuracy, 0.8)

if __name__ == "__main__":
    unittest.main()
