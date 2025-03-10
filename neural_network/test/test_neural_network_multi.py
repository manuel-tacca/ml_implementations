import numpy as np
import unittest
from neural_network.src.neural_network import MyNeuralNetwork
from data.get_data import get_multiclass_classification_data

class TestNeuralNetworkMulti(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X, self.y = get_multiclass_classification_data(n_samples=200, n_features=4, n_classes=3)
        self.model = MyNeuralNetwork(layer_sizes=[4, 5, 3], activations=['relu', 'softmax']) 

    def test_initialization(self):
        self.assertEqual(len(self.model.W_list), 2)
        self.assertEqual(len(self.model.b_list), 2)
        self.assertEqual(self.model.W_list[0].shape, (4, 5))
        self.assertEqual(self.model.W_list[1].shape, (5, 3))
        self.assertEqual(self.model.b_list[0].shape, (1, 5))
        self.assertEqual(self.model.b_list[1].shape, (1, 3))

    def test_training(self):
        initial_weights = [w.copy() for w in self.model.W_list]
        self.model.train(self.X, self.y, epochs=100, learning_rate=0.1, loss_function="categorical")
        for w_old, w_new in zip(initial_weights, self.model.W_list):
            self.assertFalse(np.array_equal(w_old, w_new))

    def test_prediction_shape(self):
        self.model.train(self.X, self.y, epochs=10, learning_rate=0.1, loss_function="categorical")
        y_pred = self.model.predict(self.X)
        self.assertEqual(y_pred.shape, (self.y.shape[0], 3))

    def test_prediction_accuracy(self):
        self.model.train(self.X, self.y, epochs=200, learning_rate=0.1, loss_function="categorical")
        y_pred = self.model.predict(self.X)
        y_pred_class = np.argmax(y_pred, axis=1)
        accuracy = np.mean(y_pred_class == self.y)
        print(f"Multiclass accuracy: {accuracy}")
        self.assertGreaterEqual(accuracy, 0.8)

if __name__ == "__main__":
    unittest.main()
