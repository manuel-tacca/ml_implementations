import numpy as np

class MyNeuralNetwork:
    
    def __init__(self, layer_sizes, activations):
        """
        Args:
        layer_sizes: lista contenente il numero di neuroni per ogni layer (incluso input e output)
        activations: lista contenente la funzione di attivazione per ogni layer (incluso output)
        """
        self.W_list = []
        self.b_list = []
        self.activations = []
        self.loss_function = self.binary_cross_entropy
        
        for i in range(len(layer_sizes) - 1):
            self.W_list.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01)
            self.b_list.append(np.zeros((1, layer_sizes[i+1])))

            if activations[i] in ("sigmoid", "linear", "relu"):
                self.activations[i] = activations[i]
            else:
                raise ValueError("Activations error, you can choose only between: sigmoid, relu, linear")

    def linear(self, z):
        return z
    
    def relu(self, z):
        return np.maximum(0, z)

    def sigmoid(self, z):
        """
        Calcola la funzione di attivazione sigmoide.
        Args:
        z (ndarray): Input del livello neurale.
        Returns:
        ndarray: Output dopo l'applicazione della sigmoide.
        """
        return 1 / (1 + np.exp(-z))
    
    def get_activation(self, activation):
        if activation == "sigmoid":
            return self.sigmoid
        elif activation == "relu":
            return self.relu
        elif activation == "linear":
            return self.linear
        else:
            raise ValueError("Activations error, you can choose only between: sigmoid, relu, linear")
    
    def my_dense(self, A_in, W, b, g):
        """
        Calcola il passaggio attraverso un dense layer.
        Args:
        A_in (ndarray): Input di dimensione (m, n), con m esempi e n caratteristiche.
        W (ndarray): Matrice dei pesi di dimensione (n, j), con j unità.
        b (ndarray): Vettore dei bias di dimensione (1, j).
        g (function): Funzione di attivazione.
        Returns:
        ndarray: Output di dimensione (m, j) dopo l'attivazione.
        """
        Z = np.matmul(A_in, W) + b
        A_out = g(Z)
        return A_out
    
    def my_sequential(self, X):
        """
        Esegue il forward pass attraverso la rete.
        Args:
        X (ndarray): Input di dimensione (m, n), con m esempi e n features.
        W_list (list): Lista delle matrici dei pesi.
        b_list (list): Lista dei vettori dei bias.
        Returns:
        ndarray: Output finale della rete.
        """
        A = X
        A_list = [A]

        for i in range(len(self.W_list)):
            A = self.my_dense(A, self.W_list[i], self.b_list[i], self.get_activation(self.activations[i]))
            A_list.append(A)

        return A_list

    def binary_cross_entropy(self, y_true, y_pred):
        """
        Calcola la Binary Cross-Entropy loss.
        Args:
        y_true (ndarray): Valori reali binari (0 o 1).
        y_pred (ndarray): Probabilità predette.
        Returns:
        float: Valore della loss.
        """
        m = len(y_pred)
        loss = -np.sum(y_true*np.log(y_pred) + (1 - y_true)*np.log(1 - y_pred)) / m
        return loss

    def categorical_cross_entropy(self, y_true, y_pred):
        """
        Calcola la Categorical Cross-Entropy loss.
        Args:
        y_true (ndarray): Matrice one-hot delle etichette reali.
        y_pred (ndarray): Probabilità predette per ogni classe.
        Returns:
        float: Valore della loss.
        """
        m = y_pred.shape[0]
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.sum(y_true * np.log(y_pred)) / m
        return loss

    def calculate_loss_gradient(self, X, y_real, y_pred):
        """
        Calculate loss gradients using symbolic derivatives for backpropagation.
        """
        if self.loss_function == self.binary_cross_entropy:
            derivative = (y_pred - y_real) / (y_pred*(1-y_pred))
        elif self.loss_function == self.categorical_cross_entropy:
            derivative = (y_pred - y_real) / y_real.shape[0]
        else:
            raise ValueError("Unrecognized loss function")
        
        return derivative
    
    def get_activation_derivative(self, activation_function, A):
        """
        Compute the derivative of the activation function
        """
        if activation_function == self.sigmoid:
            dA = A * (1 - A)
        elif activation_function == self.relu:
            dA = (A > 0).astype(float)
        elif activation_function == self.linear:
            dA = 1
        else:
            raise ValueError("Unrecognized activation function")

        return dA

    def backpropagate(self, X, y_real, learning_rate=0.1):
        """
        Esegue la backpropagation per aggiornare i pesi.
        Args:
        X (ndarray): Input dei dati di addestramento.
        y (ndarray): Etichette reali.
        learning_rate (float): Tasso di apprendimento.
        Updates:
        Modifica i pesi e i bias della rete.
        """
        # forward pass
        A_list = self.my_sequential(X)

        # loss calculation
        loss = self.calculate_loss_gradient(X, y_real, A_list[-1])

        # backpropagation
        for i in range(len(self.W_list) - 1, -1, -1):

            dA = self.get_activation_derivative(self.get_activation(self.activations[i]), A_list[i+1])

            dZ = loss*dA
            dW = np.dot(A_list[i].T, dZ)
            db = np.sum(dZ, axis=0, keepdims=True)
            
            self.W_list[i] -= learning_rate * dW
            self.b_list[i] -= learning_rate * db
            
            loss = np.dot(dZ, self.W_list[i].T)
            

    def train(self, X, y, epochs=100, learning_rate=0.1, batch_size = 32, loss_function = "binary"):
        """
        Addestra la rete neurale sui dati di input.
        Args:
        X (ndarray): Input dei dati di addestramento.
        y (ndarray): Etichette reali.
        epochs (int): Numero di epoche di allenamento.
        learning_rate (float): Tasso di apprendimento.
        Updates:
        Modifica i pesi e i bias della rete.
        """

        if loss_function in ("binary", "categorical"):
            self.loss_function = self.binary_cross_entropy if loss_function=="binary" else self.categorical_cross_entropy
        else:
            raise ValueError("Activations error, you can choose only between: binary, categorical")

        m = X.shape[0]
        
        for epoch in range(epochs):
            
            permutation = np.random.permutation(m)
            X_train_shuffled = X[permutation]
            y_train_shuffled = y[permutation]
            
            for i in range(0, m, batch_size):
                X_batch = X_train_shuffled[i:i + batch_size]
                y_batch = y_train_shuffled[i:i + batch_size]

                self.backpropagate(X_batch, y_batch, learning_rate)
            
            A_list = self.my_sequential(X)
            loss = self.calculate_loss_gradient(X, y, A_list[-1])
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")

    def predict(self, X):
        """
        Effettua una previsione sui nuovi dati.
        Args:
        X (ndarray): Input dei dati da valutare.
        Returns:
        ndarray: Output predetto (probabilità o classi).
        """
        A_list = self.my_sequential(X)
        return A_list[-1]