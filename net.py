import numpy as np
from layers import *
from metrics import LossMSE
from utils import printProgressBar

class NeuralNet:

    def __init__(self, layers):
        self.layers = layers # a list of layers

    def forward(self, X) -> np.ndarray:
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, X:np.ndarray, y_true:np.ndarray, learningRate:float):
        '''
        X: input
        y_true: expected output
        learningRate: learning rate
        '''
        #TODO: fix for online case probably the error is in the computation of diff
        y = self.forward(X)
        #if second shape is 1, then we have a single output
        if y.shape[1] == 1: y = np.reshape(y, y.shape[0])
        diff = -np.subtract(y_true, y)

        for i in range(len(self.layers)):
            back_index= len(self.layers)-1-i
            if i == 0: # output layer
                self.layers[back_index].backward(diff, learningRate)
            else: # hidden layers
                self.layers[back_index].backward(self.layers[back_index+1].delta, learningRate, self.layers[back_index+1].weights)

    def train(self, X, y, ValX = None, ValY = None, learningRate = 0.001, epochs = 200, batch_size=1) -> (list, list):

        if batch_size == 1: # One sample at a time
            for epoch in range(epochs):
                for i in range(len(X)):
                    self.backward(X[i], y[i], learningRate)
                y_predicted = self.forward(X)
                loss = LossMSE(y, y_predicted)
                print("Epoch: ", epoch, "Loss: ", loss)
            return [], []

        elif batch_size==-1: # All samples at once # TODO: check this
            trainingErrors = []
            validationErrors = []
            y_predicted = self.forward(X)
            loss = LossMSE(y, y_predicted)
            trainingErrors.append(loss)
            if ValX is not None:
                y_predicted = self.forward(ValX)
                loss = LossMSE(ValY, y_predicted)
                validationErrors.append(loss)
            print("Initial Loss: ", loss)
            for epoch in range(epochs):
                self.backward(X, y, learningRate)
                if ValX is not None:
                    #val loss
                    y_predicted = self.forward(ValX)
                    loss = LossMSE(ValY, y_predicted)
                    validationErrors.append(loss)
                #tr loss
                y_predicted = self.forward(X)
                loss = LossMSE(y, y_predicted)
                trainingErrors.append(loss)
                if epoch % 10 == 0:
                    printProgressBar(epoch, epochs, prefix = 'Progress:', suffix = f'Loss : {loss}', length = 50)
            y_predicted = self.forward(X)
            loss = LossMSE(y, y_predicted)
            printProgressBar(epochs, epochs, prefix = 'Progress:', suffix = f'Loss : {loss}', length = 50)
            return trainingErrors, validationErrors




