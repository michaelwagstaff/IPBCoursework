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

nn = nn_one_layer(input_size, hidden_size, output_size) #initialise model