import numpy as np

class MyNeuralNetwork:
    
    def __init__(self):
        pass

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def my_dense(A_in, W, b, g):
        """
        Computes dense layer
        Args:
        A_in (ndarray (m,n)) : Data, m examples, n features each
        W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
        b    (ndarray (1,j)) : bias vector, j units  
        g    activation function (e.g. sigmoid, relu..)
        Returns
        A_out (tf.Tensor or ndarray (m,j)) : m examples, j units
        """
        Z = np.matmul(A_in, W) + b
        A_out = g(Z)

        return (A_out)
    

    def my_sequential(self, X, W_list, b_list):
        """
        Computes a dense layer with activation function.
        Args:
        X         (ndarray (m,n)) : Data, m examples, n features each
        W_list    list((ndarray (n,j))) : list of weight matrices, n features per unit, j units
        b_list    list((ndarray (1,j))) : list of bias vectors, j units  
        Returns
        A_out (tf.Tensor or ndarray (m,j)) : m examples, j units
        """
        A = [self.my_dense(X, W_list[0], b_list[0], self.sigmoid)]
        for i in range(1, len(W_list)):
            A.append(self.my_dense(A[i-1], W_list[i], b_list[i], self.sigmoid))

        A_out = A[-1]
        return(A_out)