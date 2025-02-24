import numpy as np

def get_classification_data(n_samples=100, n_features=2):
    np.random.seed(0)
    X = np.random.randn(n_samples, n_features)
    true_w = np.random.randn(n_features, 1)
    true_b = np.random.randn()
    y_prob = 1 / (1 + np.exp(-(np.dot(X, true_w) + true_b)))
    y = (y_prob >= 0.5).astype(int).flatten()
    return X, y

def get_regression_data(n_samples=100, n_features=2):
    np.random.seed(0)
    X = np.random.randn(n_samples, n_features)
    true_w = np.random.randn(n_features, 1)
    true_b = np.random.randn()
    y = np.dot(X, true_w) + true_b + np.random.randn(n_samples, 1) * 0.1
    return X, y.flatten()
