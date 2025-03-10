import numpy as np
import unittest
from linear_regression.src.linear_regression import LinearRegressionGD
from data.get_data import get_regression_data

class TestLinearRegression(unittest.TestCase):
    def setUp(self):
        self.X, self.y = get_regression_data(n_samples=200, n_features=2)
        self.model = LinearRegressionGD(learning_rate=0.1, n_iterations=1000)

    def test_training(self):
        self.model.fit(self.X, self.y)
        self.assertEqual(self.model.w.shape, (self.X.shape[1], 1))
        self.assertIsInstance(self.model.b, float)

    def test_prediction_mse(self):
        self.model.fit(self.X, self.y)
        y_pred = self.model.predict(self.X)
        mse = np.mean((y_pred - self.y) ** 2)/100
        self.assertLessEqual(mse, 0.5)

if __name__ == "__main__":
    unittest.main()
