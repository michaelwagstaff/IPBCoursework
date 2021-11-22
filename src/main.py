import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

def sigmoid(a):
    siga = 1/(1 + np.exp(-a))
    return siga
    

class nn_one_layer():
    def __init__(self, input_size, hidden_size, output_size):
        #define the input/output weights W1, W2
        self.W1 = 0.1 * np.random.randn(input_size, hidden_size)
        self.W2 = 0.1 * np.random.randn(hidden_size, output_size)
        
        self.f = sigmoid
    
    def forward(self, u):
        z = np.matmul(u, self.W1)
        h = self.f(z)
        v = np.matmul(h, self.W2)
        return v, h, z

input_size = "TODO"
hidden_size = "TODO"
output_size = "TODO"

# What sizes do we want here ^? How do we know what we want?

nn = nn_one_layer(input_size, hidden_size, output_size) #initialise model

def loss(preds, targets):
    loss = np.sum((preds - targets)**2)
    return 0.5 * loss

def loss_derivative(preds, targets):
    dL_dPred = preds - targets
    return dL_dPred

def sigmoid_derivative(a):
    dsigmoid_da = sigmoid(a)*(1-sigmoid(a))
    return dsigmoid_da

def backprop(W1, W2, dL_dPred, U, H, Z):
    dL_dW2 = np.matmul(H.T, dL_dPred)
    dL_dH = np.matmul(dL_dPred, W2.T)
    dL_dZ = np.multiply(sigmoid_derivative(Z), dL_dH)
    dL_dW1 = np.matmul(U.T, dL_dZ)
    
    return dL_dW1, dL_dW2