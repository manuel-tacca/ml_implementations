import numpy as np

class MyDecisionTreeClassifier:

    def __init__(self, max_depth = 100, criterion = "entropy"):
        self.tree = []
        self.max_depth = max_depth
        self.criterion = criterion

    def compute_impurity(self, y):
        """
        Computes the impurity (entropy or gini) for 
        
        Args:
        y (ndarray): Numpy array indicating whether each example at a node is
            belonging to a class
        
        Returns:
            impurity (float): Entropy at that node
            
        """
        if len(y) == 0:
            return 0.0

        p = np.bincount(y) / len(y)
        p = p[p > 0]

        if self.criterion == "entropy":
            return -np.sum(p * np.log2(p)) 
        elif self.criterion == "gini":
            return 1 - np.sum(p**2)
        else:
            raise ValueError("Criterion must be 'entropy' or 'gini'")

    def split_dataset(self, X, node_indices, feature):
        """
        Splits the data at the given node into
        left and right branches
        
        Args:
            X (ndarray):             Data matrix of shape(n_samples, n_features)
            node_indices (list):     List containing the active indices. I.e, the samples being considered at this step.
            feature (int):           Index of feature to split on
        
        Returns:
            left_indices (list):     Indices with feature value == 1
            right_indices (list):    Indices with feature value == 0
        """
        
        left_indices = []
        right_indices = []
        
        for index in node_indices:
            if X[index, feature] == 1:
                left_indices.append(index)
            else:
                right_indices.append(index)
            
        return left_indices, right_indices

    def compute_information_gain(self, X, y, node_indices, feature):  
        """
        Compute the information of splitting the node on a given feature
        
        Args:
            X (ndarray):            Data matrix of shape(n_samples, n_features)
            y (array like):         list or ndarray with n_samples containing the target variable
            node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
            feature (int):           Index of feature to split on
    
        Returns:
            cost (float):        Cost computed
        
        """    
        # Split dataset
        left_indices, right_indices = self.split_dataset(X, node_indices, feature)
        
        # Some useful variables
        X_node, y_node = X[node_indices], y[node_indices]
        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[right_indices], y[right_indices]
        
        # You need to return the following variables correctly
        information_gain = 0
        
        w_r = len(right_indices) / len(node_indices)
        w_l = len(left_indices) / len(node_indices)
        
        entropy_node = self.compute_entropy(y_node)
        entropy_right = self.compute_entropy(y_right)
        entropy_left = self.compute_entropy(y_left)
        
        information_gain = entropy_node - (w_r*entropy_right + w_l*entropy_left)
        
        return information_gain

    def get_best_split(self, X, y, node_indices):   
        """
        Returns the optimal feature and threshold value
        to split the node data 
        
        Args:
            X (ndarray):            Data matrix of shape(n_samples, n_features)
            y (array like):         list or ndarray with n_samples containing the target variable
            node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.

        Returns:
            best_feature (int):     The index of the best feature to split
        """    
        best_feature = -1
        best_ig = 0
        
        for feature_i in range(X.shape[1]):
            curr_ig = self.compute_information_gain(X, y, node_indices, feature_i)
            if curr_ig > best_ig:
                best_ig, best_feature = curr_ig, feature_i
                    
        return best_feature if best_ig > 0 else None
    
    def check_purity(self, y, node_indeces):
        p = y[node_indeces]
        return (len(np.unique(p)) == 1)

    def build_tree(self, X, y, node_indices, current_depth):
        """
        Build a tree using the recursive algorithm that split the dataset into 2 subgroups at each node.
        This function just prints the tree.
        
        Args:
            X (ndarray):            Data matrix of shape(n_samples, n_features)
            y (array like):         list or ndarray with n_samples containing the target variable
            node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
            branch_name (string):   Name of the branch. ['Root', 'Left', 'Right']
            current_depth (int):    Current depth. Parameter used during recursive call.
    
        """ 

        # Maximum depth reached - stop splitting
        if current_depth == self.max_depth:
            return np.bincount(y[node_indices]).argmax()
        
        # Pure node - stop splitting
        if self.check_purity(y, node_indices):
            return np.bincount(y[node_indices]).argmax()
    
        # Otherwise, get best split and split the data
        # Get the best feature and threshold at this node
        best_feature = self.get_best_split(X, y, node_indices)
        if best_feature is None:
            return np.bincount(y[node_indices]).argmax() # Leaf node
        
        # Split the dataset at the best feature
        left_indices, right_indices = self.split_dataset(X, node_indices, best_feature)
        return {
            "feature": best_feature,
            "left": self.build_tree(X, y, left_indices, current_depth + 1),
            "right": self.build_tree(X, y, right_indices, current_depth + 1)
        }

    def fit(self, X, y):
        root_indices = np.arange(X.shape[0])
        self.tree = self.build_tree(X, y, root_indices, self.max_depth, 0)
        
    def predict_sample(self, tree, x):
        """Predict a single sample using the tree."""
        if isinstance(tree, dict):
            feature = tree["feature"]
            branch = "left" if x[feature] == 1 else "right"
            return self.predict_sample(tree[branch], x)
        return tree 

    def predict(self, X):
        """Predict multiple samples."""
        return np.array([self.predict_sample(self.tree, x) for x in X])