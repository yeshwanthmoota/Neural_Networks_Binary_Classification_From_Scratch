import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, neurons_hidden_layers): # input comes in the shape of (m, nx)
        self.input_size = input_size # taken as a tuple
        self.neurons_hidden_layers = neurons_hidden_layers
        self.weights = []
        self.biases = []
        self.num_layers = len(self.neurons_hidden_layers) + 1 # n hidden layers + 1 output layer
        
        prev = self.input_size[1] # (no.of neurons, nx)
        for n_neurons in neurons_hidden_layers: # weights and biases for all the hidden layers
            weights = 0.01 * np.random.randn(n_neurons, prev)
            bias = np.zeros(shape=(n_neurons, 1), dtype=float)
            self.weights.append(weights)
            self.biases.append(bias)
            prev = n_neurons
        
        weights = 0.01 * np.random.randn(1, prev) # weights and biases for the output layer
        bias = np.zeros(shape=(1, 1), dtype=float)
        self.weights.append(weights)
        self.biases.append(bias)
    
    def relu(self, Z):
        return np.maximum(0, Z)
    
    def sigmoid(self, Z):
        return 1/(1 + np.exp(-Z))
    
    def forward(self, X):
        X = np.array(X).T
        # We want to cache Z, A
        cache = []
        prev = X
        for i in range(len(self.neurons_hidden_layers)):
            Z = np.dot(self.weights[i], prev) + self.biases[i]
            A = self.relu(Z)
            cache.append((Z, A))
            prev = A
        # Now prev is equal to output of the last hidden layer
        # which is the input of the outer layer
        Z = np.dot(self.weights[-1], prev) + self.biases[-1]
        A = self.sigmoid(Z)
        cache.append((Z, A))
        return A, cache
    
    def backward(self, X, y, A, cache):
        # we store (dW, db) in gradients
        gradients = {}
        m = X.shape[1] # we didn't transpose the data to begin with so 1 still represents no. of samples

        # backward pass through output layer
        dZ = A - y
        dW = (1/m) * np.dot(dZ, cache[-2][1].T) # len(cache) = self.num_layers (both are the same)
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        gradients[self.num_layers] = (dW, db)
        prev_dA = np.dot(self.weights[-1].T, dZ) # prev_dA here is 'dA of Layer L-1'

        # backward pass through hidden layers
        for layer in range(self.num_layers - 2, -1, -1):
            # if no.of layers is 4 i.e. 3 hidden layers=> 0 1 2 and 3 being output layer which we passed now we want index 2 hence ->self.num_layers - 2
            # g'(Z) for ReLU is 1 if x>0 and 0 if x<0
            dZ = np.multiply(prev_dA, np.int64(cache[layer][0] > 0)) # np.multiply for element-wise multiplication
            dW = (1/m) * np.dot(dZ, cache[layer-1][1].T)
            db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
            gradients[layer+1] = (dW, db) # gradients of the current layer
            prev_dA = np.dot(self.weights[layer].T, dZ)
        return gradients
    
    def update_parameters(self, gradients, learning_rate):
        for layer in range(self.num_layers): # gradients start from 1-n, where as weights start from 0-(n-1), n being the no. of layers
            self.weights[layer] -= learning_rate * gradients[layer + 1][0].astype(float)
            self.biases[layer] -= learning_rate * gradients[layer + 1][1].astype(float)
    
    def train(self, X, y, epochs, learning_rate):
        for i in range(epochs):
            A, cache = self.forward(X)
            gradients = self.backward(X, y, A, cache)
            self.update_parameters(gradients, learning_rate)
            predictions = self.predict(X)
            print("epoch", i+1, "completed...., accuracy=", self.accuracy(predictions, y))
    
    def predict(self, X): # After the NN is trained
        A, cache = self.forward(X)
        # predictions = (A > 0.5).astype(int) # .astype(int) converts boolean into int
        return A
    
    def accuracy(self, y_pred, y):
        if len(y.shape) == 1: # All values passed in as [0, 1, 1, 0, 1, 1, 0, 0 ,0, .....]
            target_classes = y
        elif len(y.shape) == 2: # All values passed in as [[1, 0], [0, 1], [0, 1], [1, 0], ...] converting to [0, 1, 1, 0, ...] format
            target_classes = np.argmax(y, axis=1)

        y_pred = np.array(y_pred).T
        predicted_classes = np.argmax(y_pred, axis=1)
        # predicted_classes == target classes gives an array of (True, False, True, True, ...)
        num_trues = np.count_nonzero( (predicted_classes == target_classes) )  
        #np.count_nonzero gets how many trues are present.
        return num_trues/len(target_classes) * 100




