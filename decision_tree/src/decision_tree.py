import numpy as np

class MyDecisionTreeClassifier:
    
    def __init__(self, max_depth=100, criterion="entropy"):
        self.tree = None
        self.max_depth = max_depth
        self.criterion = criterion

    def compute_impurity(self, y):
        """
        Computes the impurity (entropy or gini) for a given set of labels.
        """
        if len(y) == 0:
            return 0.0

        p = np.bincount(y) / len(y)
        p = p[p > 0]

        if self.criterion == "entropy":
            return -np.sum(p * np.log2(p))
        elif self.criterion == "gini":
            return 1 - np.sum(p ** 2)
        else:
            raise ValueError("Criterion must be 'entropy' or 'gini'")

    def split_dataset(self, X, node_indices, feature, threshold):
        """
        Splits the data at the given node into left and right branches.
        
        For a given feature and threshold:
            - left branch: samples where X < threshold
            - right branch: samples where X >= threshold

        Args:
            X (ndarray): Data matrix of shape (n_samples, n_features)
            node_indices (list): Indices of the samples under consideration.
            feature (int): Feature index on which to split.
            threshold (float): Threshold value for the split.

        Returns:
            left_indices (list): Indices with X[:, feature] < threshold.
            right_indices (list): Indices with X[:, feature] >= threshold.
        """
        left_indices = []
        right_indices = []
        
        for index in node_indices:
            if X[index, feature] < threshold:
                left_indices.append(index)
            else:
                right_indices.append(index)
            
        return left_indices, right_indices

    def compute_information_gain(self, X, y, node_indices, feature, threshold):
        """
        Compute the information gain from splitting the node on the given feature
        and threshold.
        """
        left_indices, right_indices = self.split_dataset(X, node_indices, feature, threshold)
        
        y_node = y[node_indices]
        y_left = y[left_indices]
        y_right = y[right_indices]
        
        H_node = self.compute_impurity(y_node)
        w_left = len(y_left) / len(y_node)
        w_right = len(y_right) / len(y_node)
        H_after = w_left * self.compute_impurity(y_left) + w_right * self.compute_impurity(y_right)
        
        information_gain = H_node - H_after
        return information_gain

    def get_best_split(self, X, y, node_indices):
        """
        Finds the best feature and threshold to split the node.
        
        Returns:
            best_feature (int): Index of the best feature to split.
            best_threshold (float): Best threshold value for the split.
            If no split yields an information gain > 0, returns (None, None).
        """
        best_feature = None
        best_threshold = None
        best_ig = 0
        
        # Iterate over each feature
        for feature in range(X.shape[1]):

            feature_values = np.unique(X[node_indices, feature])
            if len(feature_values) <= 1:
                continue
            
            candidate_thresholds = (feature_values[:-1] + feature_values[1:]) / 2.0

            for threshold in candidate_thresholds:
                ig = self.compute_information_gain(X, y, node_indices, feature, threshold)
                if ig > best_ig:
                    best_ig = ig
                    best_feature = feature
                    best_threshold = threshold
                    
        return (best_feature, best_threshold) if best_ig > 0 else (None, None)
    
    def check_purity(self, y, node_indices):
        """Check if all samples in node_indices belong to the same class."""
        p = y[node_indices]
        return (len(np.unique(p)) == 1)

    def build_tree(self, X, y, node_indices, current_depth):
        """
        Recursively builds the decision tree.
        """
        if current_depth == self.max_depth or self.check_purity(y, node_indices):
            return np.bincount(y[node_indices]).argmax()

        best_feature, best_threshold = self.get_best_split(X, y, node_indices)
        if best_feature is None:
            return np.bincount(y[node_indices]).argmax()

        left_indices, right_indices = self.split_dataset(X, node_indices, best_feature, best_threshold)
        return {
            "feature": best_feature,
            "threshold": best_threshold,
            "left": self.build_tree(X, y, left_indices, current_depth + 1),
            "right": self.build_tree(X, y, right_indices, current_depth + 1)
        }

    def fit(self, X, y):
        """Fit the decision tree to the data."""
        root_indices = np.arange(X.shape[0])
        self.tree = self.build_tree(X, y, root_indices, 0)
        
    def predict_sample(self, tree, x):
        """Predict the class of a single sample x using the tree."""
        if isinstance(tree, dict):
            feature = tree["feature"]
            threshold = tree["threshold"]
            if x[feature] < threshold:
                return self.predict_sample(tree["left"], x)
            else:
                return self.predict_sample(tree["right"], x)
        return tree  # Leaf node: majority class label

    def predict(self, X):
        """Predict multiple samples."""
        return np.array([self.predict_sample(self.tree, x) for x in X])
