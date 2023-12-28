import numpy as np
from layers import *
from loss import LossMSE

class NeuralNet:

    def __init__(self, layers):
        self.layers = layers # a list of layers

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, X, y_true, learningRate):
        
        y = self.forward(X)
        diff = np.subtract(y, y_true).T # TODO: check this

        for i in range(len(self.layers)):
            back_index= len(self.layers)-1-i
            if i == 0: # output layer
                self.layers[back_index].backward(diff, learningRate)
            else: # hidden layers
                self.layers[back_index].backward(self.layers[back_index+1].delta, learningRate, self.layers[back_index+1].weights)

    def train(self, X, y, learningRate, epochs, batch_size=1):

        if batch_size == 1: # One sample at a time
            for epoch in range(epochs):
                for i in range(len(X)):
                    self.backward(X[i], y[i], learningRate)
                y_predicted = self.forward(X)
                loss = LossMSE(y, y_predicted)
                print("Epoch: ", epoch, "Loss: ", loss)

        elif batch_size==-1: # All samples at once # TODO: check this
            for epoch in range(epochs):
                self.backward(X, y, learningRate)
                y_predicted = self.forward(X)
                loss = LossMSE(y, y_predicted)
                print("Epoch: ", epoch, "Loss: ", loss)




