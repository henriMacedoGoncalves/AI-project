import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax # use built-in function to avoid numerical instability

def one_hot(x, k, dtype = np.float32):
    """Create a one-hot encoding of x of size k"""
    return np.array(x[:, None] == np.arange(k), dtype)

# Load Data
df = pd.read_csv('synthetic.csv')
x = df.iloc[:, :-1].values  # Features
y = df.iloc[:, -1].values   # Labels

# Normalize
x /= x.max()

# One-hot encode labels
num_labels = 4

#onehot_encoder = OneHotEncoder(sparse_output=False, categories='auto')

#y_new = one_hot(y.astype('int32'), 4)

y = pd.get_dummies(y)
y_new = y.values

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y_new, test_size=0.2, random_state=42)

# Further split training data for validation
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15, random_state=42)


# Verify shapes
print("Shapes:")
print("data: ", df.shape)
print("x_train:", x_train.shape)
print("y_train:", y_train.shape)
print("x_val:", x_val.shape)
print("y_val:", y_val.shape)
print("x_test:", x_test.shape)
print("y_test:", y_test.shape)

print("Training data: {} {}".format(x_train.shape, y_train.shape))
print("Test data: {} {}".format(x_test.shape, y_test.shape))


def matmul(a, b):
    """chunked matmul which converts datatypes and filters values too large to 127"""
    c = np.empty((a.shape[0], b.shape[1]), dtype = np.int8) # output
    for i in range(a.shape[0]): # iterate over rows in a 
        aa = a[i].astype(np.int32) # convert one row to extended datatype
        cc = aa @ b # broadcasting means cc is the dtype of aa 
        cc[cc > 127] = 127 # set all future overflows to 127
        # print(cc)
        c[i] = cc.astype(np.int32) # convert dtype back 
    return c

class DeepNeuralNetwork():
    def __init__(self, sizes, activation = 'tanh'):
        self.sizes = sizes
        self.length = len(sizes)
        self.best_loss = float('inf')
        self.patience = 4
        self.best_params = None
        self.test_size = 460
        # Training
        #self.f1_score = 0
        self.acc = 0
        #self.recall = 0
        #self.Precision = 0
        
        # Activation Function
        if activation == 'relu':
            self.activation = self.relu
        elif activation == 'tanh':
            self.activation = self.tanh
        else:
            raise ValueError("Activation function is not supported, please use 'relu' or 'tanh'")
        
        # Save all weights
        self.params = self.initialize()
        # Save all intermediate values i.e. activations
        self.cache = {}
        
    def relu(self, x, derivative=False):
        '''
            Derivative of ReLU is a bit more complicated since it is not differentiable at x = 0
        
            Forward path:
            relu(x) = max(0, x)
            In other word,
            relu(x) = 0, if x < 0
                    = x, if x >= 0

            Backward path:
            ∇relu(x) = 0, if x < 0
                     = 1, if x >=0
        '''
        #if derivative:
        #    x = np.where(x < 0, 0, x)
        #    x = np.where(x >= 0, 1, x)
        #    return x
        if derivative:
            return np.where(x < 0, 0, 1)
        return np.maximum(0, x)
    
    def tanh(self, x, derivative=False):
        """
        Z : non activated outputs
        Returns (A : 2d ndarray of activated outputs, df: derivative component wise)
        """
        A = np.empty(x.shape)
        A = 2.0/(1 + np.exp(-2.0*x)) - 1 
        if derivative:
            return 1 - np.square(A)
        return A
    
    
    def softmax(self, Z):
        return softmax(Z, axis=0)
    
    def initialize(self):
        # Number of nodes per layer
        # verify if layers length is 3 or 2
        if(self.length == 2):
            input_layer = 14

            hidden_layer_1 = self.sizes[0]
            hidden_layer_2 = self.sizes[1]
            output_layer = 4
            
            params = {
                "W1": np.random.randn(hidden_layer_1, input_layer) * np.sqrt(1./input_layer),
                "b1": np.zeros((hidden_layer_1, 1)) * np.sqrt(1./input_layer),
                "W2": np.random.randn(hidden_layer_2, hidden_layer_1) * np.sqrt(1./hidden_layer_1),
                "b2": np.zeros((hidden_layer_2, 1)) * np.sqrt(1./hidden_layer_1),
                "W3": np.random.randn(output_layer, hidden_layer_2) * np.sqrt(hidden_layer_2),
                "b3": np.zeros((output_layer, 1)) * np.sqrt(hidden_layer_2),
                
            }
            
        elif(self.length == 3):
            input_layer = 14
            hidden_layer_1 = self.sizes[0]
            hidden_layer_2 = self.sizes[1]
            hidden_layer_3 = self.sizes[2]
            output_layer = 4
            
            params = {
                "W1": np.random.randn(hidden_layer_1, input_layer) * np.sqrt(1./input_layer),
                "b1": np.zeros((hidden_layer_1, 1)) * np.sqrt(1./input_layer),
                "W2": np.random.randn(hidden_layer_2, hidden_layer_1) * np.sqrt(1./hidden_layer_1),
                "b2": np.zeros((hidden_layer_2, 1)) * np.sqrt(1./hidden_layer_1),
                "W3": np.random.randn(hidden_layer_3, hidden_layer_2) * np.sqrt(hidden_layer_2),
                "b3": np.zeros((hidden_layer_3, 1)) * np.sqrt(hidden_layer_2),
                "W4": np.random.randn(output_layer, hidden_layer_3) * np.sqrt(hidden_layer_3),
                "b4": np.zeros((output_layer, 1)) * np.sqrt(hidden_layer_3),
            }
        else:
            raise ValueError("Neuron Structure not supported, please use a length of either 3 or 2")
            
        return params
    
    def feed_forward(self, x):
        """
        y = tanh(wX + b)
        """
        
        if(self.length == 2):
            self.cache["X"] = x
            self.cache["Z1"] = np.dot(self.params["W1"], self.cache["X"].T) + self.params["b1"]
            #print("Shape of Z1: ", self.cache["Z1"].shape)
            #if(activation == 'relu'):
            #    print("Z1 avant activation ReLu: ", self.cache["Z1"])
            self.cache["A1"] = self.activation(self.cache["Z1"])
            #if(activation == 'relu'):
            #    print("A1 apres activation ReLu: ", self.cache["A1"])
            self.cache["Z2"] = np.dot(self.params["W2"], self.cache["A1"]) + self.params["b2"]

            self.cache["A2"] = self.activation(self.cache["Z2"])
            self.cache["Z3"] = np.dot(self.params["W3"], self.cache["A2"]) + self.params["b3"]

            self.cache["A3"] = self.softmax(self.cache["Z3"])
            
            return self.cache["A3"]
            
        elif(self.length == 3):
            self.cache["X"] = x
            self.cache["Z1"] = np.dot(self.params["W1"], self.cache["X"].T) + self.params["b1"]
            self.cache["A1"] = self.activation(self.cache["Z1"])
            self.cache["Z2"] = np.dot(self.params["W2"], self.cache["A1"]) + self.params["b2"]
            self.cache["A2"] = self.activation(self.cache["Z2"])
            self.cache["Z3"] = np.dot(self.params["W3"], self.cache["A2"]) + self.params["b3"]
            self.cache["A3"] = self.activation(self.cache["Z3"])
            self.cache["Z4"] = np.dot(self.params["W4"], self.cache["A3"]) + self.params["b4"]
            self.cache["A4"] = self.softmax(self.cache["Z4"])
            
            return self.cache["A4"]
        
        else:
            raise ValueError("Neuron Structure not supported, please use a length of either 3 or 2")
        
    
    
    def back_propagate(self, y, output):
        current_batch_size = y.shape[0]
        
        if(self.length == 2):
            if(activation == 'tanh'):
                print("Output: ", output)
                print("y: ", y)
            dZ3 = output - y.T
            dW3 = (1./current_batch_size) * matmul(dZ3, self.cache["A2"].T)
            db3 = (1./current_batch_size) * np.sum(dZ3, axis = 1, keepdims = True)
            
            # Second hidden layer gradients
            # dA2 = np.matmul(self.params["W3"].T, dZ3)
            dA2 = self.params["W3"].T @ dZ3
            dZ2 = dA2 * self.activation(self.cache["Z2"], derivative = True)
            dW2 = (1./current_batch_size) * matmul(dZ2, self.cache["A1"].T)
            db2 = (1./current_batch_size) * np.sum(dZ2, axis = 1, keepdims = True)
            
            # First Hidden Layer gradients
            # dA1 = np.matmul(self.params["W2"].T, dZ2)
            dA1 = self.params["W2"].T @ dZ2
            dZ1 = dA1 * self.activation(self.cache["Z1"], derivative = True)
            dW1 = (1./current_batch_size) * (dZ1 @ self.cache["X"])
            db1 = (1./current_batch_size) * np.sum(dZ1, axis = 1, keepdims = True)
            
            self.grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}
            return self.grads
            
        elif(self.length == 3):
            dZ4 = output - y.T
            dW4 = (1./current_batch_size) * (dZ4 @ self.cache["A3"].T)
            db4 = (1./current_batch_size) * np.sum(dZ4, axis = 1, keepdims = True)
            
            # Third hidden layer gradients
            dA3 = np.matmul(self.params["W4"].T, dZ4)
            dZ3 = dA3 * self.activation(self.cache["Z3"], derivative = True)
            ## Verificar utilización de A2
            dW3 = (1./current_batch_size) * (dZ3 @ self.cache["A2"].T)
            db3 = (1./current_batch_size) * np.sum(dZ3, axis = 1, keepdims = True)
            
            # Second hidden layer gradients
            dA2 = np.matmul(self.params["W3"].T, dZ3)
            dZ2 = dA2 * self.activation(self.cache["Z2"], derivative = True)
            dW2 = (1./current_batch_size) * (dZ2 @ self.cache["A1"].T)
            db2 = (1./current_batch_size) * np.sum(dZ2, axis = 1, keepdims = True)
            
            # First Hidden Layer gradients
            dA1 = np.matmul(self.params["W2"].T, dZ2)
            dZ1 = dA1 * self.activation(self.cache["Z1"], derivative = True)
            dW1 = (1./current_batch_size) * (dZ1 @ self.cache["X"])
            db1 = (1./current_batch_size) * np.sum(dZ1, axis = 1, keepdims = True)
            
            self.grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3, "W4": dW4, "b4": db4}
            return self.grads

        else:
            raise ValueError("Neuron Structure not supported, please use a length of either 3 or 2")
            
        # return self.grads
    
    def cross_entropy_loss(self, y, output):
        '''
            L(y, ŷ) = −∑ylog(ŷ).
        '''
        #l_sum = np.sum(np.multiply(y.T, np.log(output)))
        #m = y.shape[0]
        #l = -(1./m) * l_sum
        #return l
        epsilon = 1e-10  # Small value to ensure numerical stability
        m = y.shape[0]
        l_sum = np.sum(np.multiply(y.T, np.log(output + epsilon)))  # Add epsilon to prevent log(0)
        loss = -(1. / m) * l_sum
        return loss
    
    def optimize(self, l_rate=0.1):
        """
        Stochatic Gradient Descent (SGD):
            θ^(t+1) <- θ^t - η∇L(y, ŷ)
        """
        for key in self.params:
            #print("Before: ", self.params[key])
            #np.add(self.params[key], (l_rate * self.grads[key]))
            self.params[key] = self.params[key] - (l_rate * self.grads[key])
            #print("After: ", self.params[key])
            
    def accuracy(self, y, output):
        y_pred = np.argmax(output.T, axis=-1)
        #print("Predicted values:", y_pred)
        y_true = np.argmax(y, axis=-1)
        #print("Actual values:", y_true)
        # Calculate True Positives, True Negatives, False Positives, False Negatives
        TP = np.sum((y_true == 1) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))

        # Calculate accuracy
        if((TP + TN + FP + FN) == 0):
            accuracy = 0
        else:
            accuracy = (TP + TN) / (TP + TN + FP + FN)

        # Precision
        #if((TP + FP) == 0):
        #    precision = 0
        #else:
        #    precision = TP / (TP + FP)

        # Recall
        if((TP + FN)==0):
            recall = 0
        else:
            recall = TP / (TP + FN)

        # F1-score
        #if((precision + recall) == 0):
        #    f1_score = 0
        #else:
        #    f1_score = 2 * (precision * recall) / (precision + recall)
        
        return accuracy#, precision, recall, f1_score
    
    def precsision(self, y, output):
        y_pred = np.argmax(output.T, axis=-1)
        #print("Predicted values:", y_pred)
        y_true = np.argmax(y, axis=-1)
        #print("Actual values:", y_true)
        # Calculate True Positives, True Negatives, False Positives, False Negatives
        TP = np.sum((y_true == 1) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        
        # Precision
        if((TP + FP) == 0):
            precision = 0
        else:
            precision = TP / (TP + FP)
        return precision
        
        
        

    def train(self, x_train, y_train, x_test, y_test,x_val, y_val, epochs=100, batch_size = 64, l_rate=0.1):
        
        self.epochs = epochs
        self.batch_size = batch_size
        num_batches = 390
        
        # Initialize optimizer
        #self.optimizer = optimizer
        
        start_time = time.time()
        template = "Epoch {}: {:.2f}s, train acc={:.2f}, train loss={:.2f}, test acc={:.2f}, test loss={:.2f}"
        #template2 = "train F1 Score = {:.2f}, train Precision={:.2f},  train Recall={:.2f}, test F1 Score = {:.2f}, test Precision={:.2f},  test Recall={:.2f},"
        
        best_epoch = 0
        patience_count = 0
        # Train
        for i in range(self.epochs):
            # Shuffle
            permutation = np.random.permutation(x_train.shape[0])
            x_train_shuffled = x_train[permutation]
            y_train_shuffled = y_train[permutation]
            
            
            for j in range(num_batches):
                # Batch
                begin = j* self.batch_size
                end = min(begin + self.batch_size, x_train.shape[0]-1)
                x = x_train_shuffled[begin:end]
                y = y_train_shuffled[begin:end]
                
                # Forward
                output = self.feed_forward(x)
                #print("Predicted: ", np.argmax(output, axis=-1))
                #print("Actual: ", np.argmax(y, axis=-1))
                # Backprop
                grad = self.back_propagate(y, output)
                # Optimize
                self.optimize(l_rate = l_rate)
            
            # At this point, the weights are supposed to be optimized and thus, we measure with the data
            
            # Evaluate performance
            # Training data
            train_loss = 0
            train_acc = 0
            test_acc = 0
            test_loss = 0
            print("After stochastic Gradient Descent:")
            for j in range(num_batches):
                # Batch
                begin = j* self.batch_size
                end = min(begin + self.batch_size, x_train.shape[0]-1)
                x = x_train_shuffled[begin:end]
                y = y_train_shuffled[begin:end]
                train_output = self.feed_forward(x)
                #zeros = 0
                #for i in np.argmax(train_output, axis=-1):
                #    if(np.argmax(train_output, axis=-1)[i] == 0):
                #        zeros += 1
                #if(zeros == 4):
                ##    print("Output that only predicted 0s: ", train_output.T)
                 #   print("Predicted: ", np.argmax(train_output, axis=-1))
                #print("Predicted: ", np.argmax(train_output, axis=-1))
                #print("Actual: ", np.argmax(y, axis=-1))
                train_acc += self.accuracy(y, train_output)
                train_loss += self.cross_entropy_loss(y, train_output)
            train_acc /= num_batches
            train_loss /= num_batches
            
            #for j in range(self.test_size):
                # Batch
            #    begin = j* self.batch_size
            #    end = min(begin + self.batch_size, x_train.shape[0]-1)
            #    x = x_test[begin:end]
            #    y = y_test[begin:end]
            #    test_output = self.feed_forward(x)
                #print("Predicted: ", np.argmax(train_output, axis=-1))
                #print("Actual: ", np.argmax(y, axis=-1))
            #    test_acc += self.accuracy(y, test_output)
            #    test_loss += self.cross_entropy_loss(y, test_output)
            #test_acc /= self.test_size
            #test_acc /= self.test_size

            #train_output = self.feed_forward(x_train)
            #print("Predicted (Training data): ", np.argmax(train_output.T, axis=-1))
            #print("Actual (Training data): ", np.argmax(y_train[0], axis=-1))
            #train_acc, train_prec, train_recall, train_f1score = self.accuracy(y_train, train_output)
            #train_acc = self.accuracy(y_train, train_output)
            #train_loss = self.cross_entropy_loss(y_train, train_output)
            
            # Validation
            val_output = self.feed_forward(x_val)
            #print("Validation Set: ",np.argmax(y_val, axis=-1))
            val_loss = self.cross_entropy_loss(y_val, val_output)
            #print("Predicted: ",np.argmax(val_output.T, axis=-1))
            
            # Test data
            
            # accuracy, precision, recall, f1_score
            test_output = self.feed_forward(x_test)
            # test_acc, test_prec, test_recall, test_f1score = self.accuracy(y_test, test_output)
            test_acc = self.accuracy(y_test, test_output)
            test_loss = self.cross_entropy_loss(y_test, test_output)
            
            print(template.format(i+1, time.time()-start_time, train_acc, train_loss, test_acc, test_loss))
            #print(template2.format(train_f1score, train_prec, train_recall, test_f1score, test_prec, test_recall))
            
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_params = self.params.copy
                best_epoch = i
                patience_count = 0
            else:
                patience_count += 1
            
            # Early stopping
            if patience_count >= self.patience:
                print("Early stopping at epoch:", i+1)
                print("Best validation loss:", self.best_loss)
                print("Best epoch:", best_epoch+1)
                self.params = self.best_params
                #self.f1_score = train_f1score
                self.acc = train_acc
                #self.recall = train_recall
                #self.Precision = train_prec
                
                break
                
        
        return test_output, val_loss

                
            
                
            
# Define architectures and activations
# architectures = [(10, 8, 6), (10, 8, 4), (6, 4)]
architectures = [(6, 4), (10, 8, 6), (10, 8, 4)]
activations = ['relu', 'tanh']

best_models = { 'relu': [], 'tanh': []}

# Train and evaluate models
for activation in activations:
    for architecture in architectures:
        
        print(f"Architecture: {architecture}, Activation: {activation}")
            
        # Create and train the model
        model = DeepNeuralNetwork(architecture, activation=activation)
        # Evaluate the model on the test set
        
        test_output, best_val_loss = model.train(x_train, y_train, x_test, y_test, x_val, y_val, epochs=100, batch_size=4, l_rate=0.1)
        # Keep track of the model's predictions and test loss
        best_models[activation].append((test_output.T, best_val_loss))
        
# TODO Sort the models by lowest val loss
# Show best Model by activation function

