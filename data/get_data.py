import numpy as np

def get_classification_data(n_samples=100, n_features=2, n_classes=2):
    np.random.seed(0)
    X = np.random.randn(n_samples, n_features)
    true_w = np.random.randn(n_features, n_classes)
    true_b = np.random.randn(n_classes)
    y_prob = np.exp(np.dot(X, true_w) + true_b)
    y_prob /= np.sum(y_prob, axis=1, keepdims=True)
    y = np.argmax(y_prob, axis=1)
    return X, y

def get_regression_data(n_samples=100, n_features=2):
    np.random.seed(0)
    X = np.random.randn(n_samples, n_features)
    true_w = np.random.randn(n_features, 1)
    true_b = np.random.randn()
    y = np.dot(X, true_w) + true_b + np.random.randn(n_samples, 1) * 0.1
    return X, y.flatten()

def get_multiclass_classification_data(n_samples=100, n_features=2, n_classes=3):
    np.random.seed(0)
    X = np.random.randn(n_samples, n_features)
    true_w = np.random.randn(n_features, n_classes)
    true_b = np.random.randn(n_classes)
    y_prob = np.exp(np.dot(X, true_w) + true_b)
    y_prob /= np.sum(y_prob, axis=1, keepdims=True)
    y = np.argmax(y_prob, axis=1)
    return X, y
