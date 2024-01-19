import numpy as np
from layers import *
from metrics import LossMSE
from utilities import printProgressBar

class NeuralNet:

    def __init__(self, layers):
        self.layers = layers # a list of layers
    
    def reset(self):
        for layer in self.layers:
            layer.reset()

    def forward(self, X) -> np.ndarray:
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, X:np.ndarray, y_true:np.ndarray, learningRate:float, lambdaRegularization:float = 0, momentum:float = 0):
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
                self.layers[back_index].backward(diff, learningRate, None, lambdaRegularization)
            else: # hidden layers
                self.layers[back_index].backward(self.layers[back_index+1].delta, learningRate, self.layers[back_index+1].weights, lambdaRegularization)

    def train(self, X, y, ValX = None, ValY = None, learningRate = 0.001, epochs = 200, batch_size=1, lambdaRegularization:float = 0, momentum:float = 0, patience:int = -1) -> (list, list):

        trainingErrors = []
        validationErrors = []
        trainingErrors.append(self.get_errors(X, y, LossMSE))

        if ValX is not None:
            validationErrors.append(self.get_errors(ValX, ValY, LossMSE))

        print("Initial loss: ", trainingErrors[0])
        
        if batch_size==-1: # All samples at once (batch)

            for epoch in range(epochs):
                self.backward(X, y, learningRate, lambdaRegularization, momentum)
                if ValX is not None:
                    #val loss
                    validationErrors.append(self.get_errors(ValX, ValY, LossMSE))
                #tr loss
                trainingErrors.append(self.get_errors(X, y, LossMSE))
                if epoch % 10 == 0:
                    printProgressBar(epoch, epochs, prefix = 'Progress:', suffix = f'Loss : {trainingErrors[epoch+1]}', length = 50)
                
                # check stopping criteria
                if self.check_stopping_criteria(validationErrors, trainingErrors, lambdaRegularization, patience):
                    print("\nEarly stopping at epoch: ", epoch)
                    return trainingErrors, validationErrors

            printProgressBar(epochs, epochs, prefix = 'Progress:', suffix = f'Loss : {trainingErrors[epochs]}', length = 50)
            return trainingErrors, validationErrors

        else: # Mini-batch, we can also do online training by setting batch_size=1

            for epoch in range(epochs):
                #shuffle data
                p = np.random.permutation(len(X))
                X = X[p]
                y = y[p]
                for i in range(0, len(X), batch_size):
                    if i+batch_size < len(X):
                        self.backward(X[i:i+batch_size], y[i:i+batch_size], learningRate, lambdaRegularization, momentum)
                    else:
                        self.backward(X[i:], y[i:], learningRate, lambdaRegularization)
                if ValX is not None:
                    #val loss
                    validationErrors.append(self.get_errors(ValX, ValY, LossMSE))
                #tr loss
                trainingErrors.append(self.get_errors(X, y, LossMSE))
                if epoch % 10 == 0:
                    printProgressBar(epoch, epochs, prefix = 'Progress:', suffix = f'Loss : {trainingErrors[epoch+1]}', length = 50)

                # check stopping criteria
                if self.check_stopping_criteria(validationErrors, trainingErrors, lambdaRegularization, patience):
                    print("\nEarly stopping at epoch: ", epoch)
                    return trainingErrors, validationErrors
                
            printProgressBar(epochs, epochs, prefix = 'Progress:', suffix = f'Loss : {trainingErrors[epochs]}', length = 50)
            return trainingErrors, validationErrors
    
    def get_errors(self, X: np.array, y:np.array, loss:callable):
        y_predicted = self.forward(X)
        loss = loss(y, y_predicted)
        return loss
    
    def check_stopping_criteria(self, validationErrors:[float], trainingErrors:[float], lambdaRegularization:float, patience:int) -> bool:
        '''
        validationErrors: list of validation errors
        trainingErrors: list of training errors
        lambdaRegularization: regularization parameter
        patience: number of epochs to wait before stopping

        returns True if the stopping criteria is met, False otherwise
        '''
        if patience == -1 and lambdaRegularization > 0: #TODO: check theory
            if len(trainingErrors) < 10:
                return False
            #check if the training error is decreasing
            for i in range(len(trainingErrors)-2, len(trainingErrors)):
                if trainingErrors[i] < trainingErrors[i-1]:
                    return False
            return True
        
        elif patience > 0:
            if len(validationErrors) < patience:
                return False
            #check if the validation error is increasing
            for i in range(len(validationErrors)-patience, len(validationErrors)):
                #print(validationErrors[i], validationErrors[i-1], validationErrors[i] < validationErrors[i-1])
                if validationErrors[i] < validationErrors[i-1]: #if it is improving, we stop even if the validation error remains the same
                    return False
            return True
        
        return False