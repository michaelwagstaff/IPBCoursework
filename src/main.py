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

input_size = 784
hidden_size = 100
output_size = 1

# What sizes do we want here ^? How do we know what we want?

nn = nn_one_layer(input_size, hidden_size, output_size) #initialise model

def loss_function(preds, targets):
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

def generate_batch(train_imgs, train_lbls, batch_size):
    #differentiate inputs (features) from targets and transform each into 
    #numpy array with each row as an example
    inputs = np.vstack(train_imgs)
    targets = np.vstack(train_lbls)
    
    #randomly choose batch_size many examples; note there will be
    #duplicate entries when batch_size > len(dataset) 
    rand_inds = np.random.randint(0, len(inputs), batch_size)
    inputs_batch = inputs[rand_inds]
    targets_batch = targets[rand_inds]
    
    return inputs_batch, targets_batch
    

def train_one_batch(nn, train_imgs, train_lbls, batch_size, learning_rate):
    inputs, targets = generate_batch(train_imgs, train_lbls, batch_size)
    preds, H, Z = nn.forward(inputs)

    loss = loss_function(preds, targets)

    dL_dPred = loss_derivative(preds, targets)
    dL_dW1, dL_dW2 = backprop(nn.W1, nn.W2, dL_dPred, U=inputs, H=H, Z=Z)

    nn.W1 -= learning_rate * dL_dW1
    nn.W2 -= learning_rate * dL_dW2
    
    return loss

train_one_batch(nn, train_images, train_labels, 200, 0.02)

print("Finished!")