import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import random
from functools import partial
import functools
#from sklearn.datasets import fetch_ml

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = np.true_divide(train_images, 255)
test_images = np.true_divide(test_images, 255)

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
        #print("W1: " + str(np.sum(self.W1)))
        #print("W2: " + str(np.sum(self.W2)))
        z = np.matmul(u, self.W1)
        h = self.f(z)
        v = np.matmul(h, self.W2)
        return v, h, z

input_size = 784
hidden_size = 100
output_size = 10

# What sizes do we want here ^? How do we know what we want?

nn = nn_one_layer(input_size, hidden_size, output_size) #initialise model

def loss_function(preds, targets):
    loss = np.sum((preds - targets)**2)/784
    return 0.5 * loss

def loss_derivative(preds, targets):
    dL_dPred = (preds - targets)/784
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

def generate_batch(images, labels, batch_size):
    #differentiate inputs (features) from targets and transform each into 
    #numpy array with each row as an example
    inputs = images.reshape(len(labels),784)
    targets = np.vstack(labels)
    
    #randomly choose batch_size many examples; note there will be
    #duplicate entries when batch_size > len(dataset) 
    rand_inds = np.random.randint(0, len(labels), batch_size)
    inputs_batch = inputs[rand_inds]
    targets_batch = targets[rand_inds]
    #print(inputs_batch)
    
    return inputs_batch, targets_batch
    

def train_one_batch(nn, train_imgs, train_lbls, batch_size, learning_rate):
    inputs, targets = generate_batch(train_imgs, train_lbls, batch_size)
    preds, H, Z = nn.forward(inputs)
    def f(x):
        t = np.zeros(10)
        t[x]=1
        return t
    vector_targets = list(map(f, targets)) # Why on earth do I have to convert manually to a list
                                           # I swear to god Python sucks so bad
    #print(vector_targets[0])
    loss = loss_function(preds, vector_targets)

    dL_dPred = loss_derivative(preds, vector_targets)
    dL_dW1, dL_dW2 = backprop(nn.W1, nn.W2, dL_dPred, U=inputs, H=H, Z=Z)

    nn.W1 -= learning_rate * dL_dW1
    nn.W2 -= learning_rate * dL_dW2
    
    return loss

def test(nn, test_images, test_labels):
    inputs, targets = generate_batch(test_images, test_labels, batch_size=100)
    preds, H, Z = nn.forward(inputs)
    def f(x):
        t = np.zeros(10)
        t[x]=1
        return t
    vector_targets = list(map(f, targets)) # Why can't we just use Haskell??
    loss = loss_function(preds, vector_targets)
    sum = 0
    print(preds[0])
    for i in range(0, len(targets)):
        for j in range(0,10):
            if(preds[i][j]==max(preds[i]) and vector_targets[i][j] == 1):
                sum+=1
    print(str(sum) + " of " + str(len(targets)) + " correct")
    print(loss)
    return loss

indices = [x for x in range(0,500)]
results = []
for i in range(0,10000):
    train_one_batch(nn, train_images, train_labels, 200, 0.1)
    if(i % 20 == 0):
        results.append(test(nn, test_images, test_labels))
    plt.xlabel("Test run number")
    plt.ylabel("Loss function")
plt.plot(indices, results, "x")
plt.show()



print("Finished!")