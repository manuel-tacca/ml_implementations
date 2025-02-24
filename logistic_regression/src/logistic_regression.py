import numpy as np

class LogisticRegressionGD:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.w = None
        self.b = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        m, n = X.shape
        self.w = np.random.randn(n, 1)
        self.b = np.random.randn()
        y = y.reshape(-1, 1)
        
        for _ in range(self.n_iterations):
            z = np.dot(X, self.w) + self.b
            f_wb = self.sigmoid(z)
            
            error = f_wb - y
            dj_dw = np.dot(X.T, error) / m
            dj_db = np.sum(error) / m
            
            self.w -= self.learning_rate * dj_dw
            self.b -= self.learning_rate * dj_db
    
    def predict(self, X):
        z = np.dot(X, self.w) + self.b
        f_wb = self.sigmoid(z)
        return (f_wb >= 0.5).astype(int).flatten()
