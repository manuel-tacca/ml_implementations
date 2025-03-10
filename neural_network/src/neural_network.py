import numpy as np

class MyNeuralNetwork:
    def __init__(self, layer_sizes, activations):
        self.W_list = []
        self.b_list = []
        self.activations = []
        # Default loss function
        self.loss_function = self.binary_cross_entropy

        # Initialize weights and biases
        for i in range(len(layer_sizes) - 1):
            if activations[i] not in ("sigmoid", "linear", "relu", "softmax"):
                raise ValueError("Activations error, you can choose only between: sigmoid, relu, linear, softmax")
            self.activations.append(activations[i])
            # Use He initialization for ReLU; otherwise, use small scaling.
            if activations[i] == "relu":
                W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2 / layer_sizes[i])
            else:
                W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01
            self.W_list.append(W)
            self.b_list.append(np.zeros((1, layer_sizes[i+1])))

    def linear(self, z):
        return z

    def relu(self, z):
        return np.maximum(0, z)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def softmax(self, z):
        # Numerical stability: subtract max per row
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / (np.sum(exp_z, axis=1, keepdims=True) + 1e-15)

    def get_activation(self, activation):
        if activation == "sigmoid":
            return self.sigmoid
        elif activation == "relu":
            return self.relu
        elif activation == "linear":
            return self.linear
        elif activation == "softmax":
            return self.softmax
        else:
            raise ValueError("Activations error, you can choose only between: sigmoid, relu, linear, softmax")

    def my_dense(self, A_in, W, b, activation_func):
        Z = np.dot(A_in, W) + b
        if np.isnan(Z).any():
            print("Detected NaN in forward propagation!")
            print(f"A_in stats - min: {np.min(A_in)}, max: {np.max(A_in)}, mean: {np.mean(A_in)}")
            print(f"W stats - min: {np.min(W)}, max: {np.max(W)}, mean: {np.mean(W)}")
            print(f"b stats - min: {np.min(b)}, max: {np.max(b)}, mean: {np.mean(b)}")
            raise ValueError("NaN detected in forward propagation!")
        A_out = activation_func(Z)
        return A_out

    def my_sequential(self, X):
        A = X
        A_list = [A]
        for i in range(len(self.W_list)):
            activation_func = self.get_activation(self.activations[i])
            A = self.my_dense(A, self.W_list[i], self.b_list[i], activation_func)
            A_list.append(A)
        return A_list

    def binary_cross_entropy(self, y_true, y_pred):
        epsilon = 1e-10
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / m
        return loss

    def categorical_cross_entropy(self, y_true, y_pred):
        m = y_true.shape[0]
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.sum(y_true * np.log(y_pred)) / m
        return loss

    def one_hot_encode(self, y):
        num_classes = len(np.unique(y))
        return np.eye(num_classes)[y]

    def calculate_loss_gradient(self, y_true, y_pred):
        m = y_true.shape[0]
        return (y_pred - y_true) / m

    def get_activation_derivative(self, activation_function, A):
        if activation_function == self.sigmoid:
            return A * (1 - A)
        elif activation_function == self.relu:
            return (A > 0).astype(float)
        elif activation_function == self.linear:
            return np.ones_like(A)
        elif activation_function == self.softmax:
            # Softmax derivative is handled via the loss gradient when combined with cross entropy.
            raise ValueError("Softmax derivative is handled in the loss gradient for the output layer.")
        else:
            raise ValueError("Unrecognized activation function")

    def backpropagate(self, X, y_real, learning_rate=0.01):
        A_list = self.my_sequential(X)
        m = y_real.shape[0]
        # For the output layer (using cross entropy with softmax or sigmoid), the gradient simplifies:
        dZ = (A_list[-1] - y_real) / m

        # Loop backward over layers
        for i in range(len(self.W_list) - 1, -1, -1):
            A_prev = A_list[i]
            # For hidden layers, multiply by activation derivative.
            if i != len(self.W_list) - 1:
                activation_func = self.get_activation(self.activations[i])
                dA = self.get_activation_derivative(activation_func, A_list[i+1])
                dZ = dZ * dA
            dW = np.dot(A_prev.T, dZ)
            db = np.sum(dZ, axis=0, keepdims=True)
            self.W_list[i] -= learning_rate * dW
            self.b_list[i] -= learning_rate * db
            if i > 0:
                dZ = np.dot(dZ, self.W_list[i].T)

    def train(self, X, y, epochs=100, learning_rate=0.1, loss_function="binary"):
        # If using categorical loss, ensure dimensions match:
        if loss_function == "binary":
            self.loss_function = self.binary_cross_entropy
        elif loss_function == "categorical":
            if len(np.unique(y)) == 2:
                # For binary classification, use binary cross entropy even if categorical was specified.
                self.loss_function = self.binary_cross_entropy
            else:
                self.loss_function = self.categorical_cross_entropy
                y = self.one_hot_encode(y)
        else:
            raise ValueError("Loss function must be 'binary' or 'categorical'")

        m = X.shape[0]
        for epoch in range(epochs):
            for i in range(m):
                X_sample = X[i:i+1]
                y_sample = y[i:i+1]
                self.backpropagate(X_sample, y_sample, learning_rate)
            A_list = self.my_sequential(X)
            loss = self.loss_function(y, A_list[-1])
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")

    def predict(self, X):
        A_list = self.my_sequential(X)
        output = A_list[-1]
        # If binary classification (single output unit), squeeze to 1D.
        if output.ndim == 2 and output.shape[1] == 1:
            return output.squeeze(axis=1)
        return output
