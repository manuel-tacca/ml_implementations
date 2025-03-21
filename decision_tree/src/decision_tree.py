import numpy as np

class MyDecisionTreeClassifier:

    def __init__(self, max_depth = 100):
        self.tree = []
        self.max_depth = max_depth

    def compute_entropy(self, y):
        """
        Computes the entropy for 
        
        Args:
        y (ndarray): Numpy array indicating whether each example at a node is
            edible (`1`) or poisonous (`0`)
        
        Returns:
            entropy (float): Entropy at that node
            
        """
        entropy = 0.
        
        if len(y) != 0:
            p1 = np.count_nonzero(y==1) / len(y)
        else:
            p1 = 0
        
        if p1 == 0 or p1 == 1:
            entropy = 0.0
        else:
            entropy = -p1*np.log2(p1) - (1-p1)*np.log2(1-p1)
                
        return entropy

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
                best_ig = curr_ig
                best_feature = feature_i
                    
        return best_feature
    
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
            return
        
        # Pure node - stop splitting
        if self.check_purity(y, node_indices):
            return
    
        # Otherwise, get best split and split the data
        # Get the best feature and threshold at this node
        best_feature = self.get_best_split(X, y, node_indices)
        
        # Split the dataset at the best feature
        left_indices, right_indices = self.split_dataset(X, node_indices, best_feature)
        self.tree.append((left_indices, right_indices, best_feature))
        
        # continue splitting the left and the right child. Increment current depth
        self.build_tree(X, y, left_indices, self.max_depth, current_depth+1)
        self.build_tree(X, y, right_indices, self.max_depth, current_depth+1)

    # def one_hot_encode(self, X, y):
    #     # for each feature I need every single example's value
    #     for feature_i in range(X.shape[1]):
    #         curr_values = X[:, feature_i]
    #         code = np.unique(curr_values)
    #         ...

    def fit(self, X, y):

        # self.one_hot_encode(X, y)

        root_indices = np.array(range(X.shape[1]))
        self.build_tree(X, y, root_indices, self.max_depth, 0)
        print(self.tree)
