import os
import numpy as np
from layers import *
from metrics import LossMSE
from r_prop_parameter import RProp
from utilities import printProgressBar

class NeuralNet:

    def __init__(self, layers:[LayerDense]):
        self.layers = layers # a list of layers
    

    def reset(self):
        for layer in self.layers:
            layer.reset()


    def forward(self, X) -> np.ndarray:
        for layer in self.layers:
            X = layer.forward(X)
        return X


    def backward(self, X:np.ndarray, y_true:np.ndarray, learningRate:float, lambdaRegularization:float = 0, momentum:float = 0, r_prop:RProp|None = None):
        '''
        X: input
        y_true: expected output
        learningRate: learning rate
        '''
        y = self.forward(X)
        #if second shape is 1, then we have a single output
        if y.shape[1] == 1: y = np.reshape(y, y.shape[0])
        diff = -np.subtract(y_true, y)
        #normalize diff
        diff = diff/len(X)
        for i in range(len(self.layers)):
            back_index= len(self.layers)-1-i
            if i == 0: # output layer
                self.layers[back_index].backward(diff, learningRate, None, lambdaRegularization, momentum, r_prop)
            else: # hidden layers
                self.layers[back_index].backward(self.layers[back_index+1].delta, learningRate, self.layers[back_index+1].weights, lambdaRegularization, momentum, r_prop)


    def train(self, X, y, ValX = None, ValY = None, learningRate = 0.001, epochs = 200, batch_size=-1, lambdaRegularization:float = 0, momentum:float = 0, patience:int = -1, tau:int=0, r_prop:RProp|None = None, accuracy:callable = None) -> (list, list):

        trainingErrors = []
        validationErrors = []
        trainingAccuracy = []
        validationAccuracy = []
        self.update_metrics(X, y, LossMSE, accuracy, trainingErrors, trainingAccuracy)

        if ValX is not None:
            self.update_metrics(ValX, ValY, LossMSE, accuracy, validationErrors, validationAccuracy)

        print("Initial loss: ", trainingErrors[0])
        
        if batch_size==-1: # All samples at once (batch)
            if tau != 0:
                raise Exception("Variable learning rate is not supported for batch training")
            for epoch in range(epochs):
                self.backward(X, y, learningRate, lambdaRegularization, momentum, r_prop)
                if ValX is not None:
                    #validationErrors.append(self.get_errors(ValX, ValY, LossMSE))
                    self.update_metrics(ValX, ValY, LossMSE, accuracy, validationErrors, validationAccuracy)
                #trainingErrors.append(self.get_errors(X, y, LossMSE))
                self.update_metrics(X, y, LossMSE, accuracy, trainingErrors, trainingAccuracy)
                printProgressBar(epoch, epochs, prefix = 'Progress:', suffix = f'Loss : {trainingErrors[epoch+1]}', length = 50)
                # check stopping criteria
                if self.check_stopping_criteria(validationErrors, trainingErrors, lambdaRegularization, patience):
                    print("\nEarly stopping at epoch: ", epoch)
                    return trainingErrors, validationErrors, trainingAccuracy, validationAccuracy

            printProgressBar(epochs, epochs, prefix = 'Progress:', suffix = f'Loss : {trainingErrors[epochs]}', length = 50)
            return trainingErrors, validationErrors, trainingAccuracy, validationAccuracy

        else: # Mini-batch, we can also do online training by setting batch_size=1
            if r_prop:
                raise Exception("RProp is not supported for mini-batch/onlines training")
            for epoch in range(epochs):
                #shuffle data
                p = np.random.permutation(len(X))
                X = X[p]
                y = y[p]
                for i in range(0, len(X), batch_size):
                    if i+batch_size < len(X):
                        self.backward(X[i:i+batch_size], y[i:i+batch_size], self.variable_learning_rate(epoch, learningRate, tau), lambdaRegularization, momentum)
                    else:
                        self.backward(X[i:], y[i:], self.variable_learning_rate(epoch, learningRate, tau), lambdaRegularization, momentum)
                if ValX is not None:
                    self.update_metrics(ValX, ValY, LossMSE, accuracy, validationErrors, validationAccuracy)
                self.update_metrics(X, y, LossMSE, accuracy, trainingErrors, trainingAccuracy)
                #if epoch % 10 == 0:
                printProgressBar(epoch, epochs, prefix = 'Progress:', suffix = f'Loss : {trainingErrors[epoch+1]}', length = 50)

                # check stopping criteria
                if self.check_stopping_criteria(validationErrors, trainingErrors, lambdaRegularization, patience):
                    print("\nEarly stopping at epoch: ", epoch)
                    return trainingErrors, validationErrors, trainingAccuracy, validationAccuracy
                
            printProgressBar(epochs, epochs, prefix = 'Progress:', suffix = f'Loss : {trainingErrors[epochs]}', length = 50)
            return trainingErrors, validationErrors, trainingAccuracy, validationAccuracy
    

    def get_errors(self, X: np.array, y:np.array, loss:callable):
        y_predicted = self.forward(X)
        loss = loss(y, y_predicted)
        return loss
    
    def update_metrics(self, X: np.array, y:np.array, loss:callable, accuracy:callable, errors:list, accuracies:list):
        y_predicted = self.forward(X)
        loss = loss(y, y_predicted)
        errors.append(loss)
        if accuracy is not None:
            acc = accuracy(y, y_predicted)
            accuracies.append(acc)
        
    

    def check_stopping_criteria(self, validationErrors:[float], trainingErrors:[float], lambdaRegularization:float, patience:int) -> bool:
        '''
        validationErrors: list of validation errors
        trainingErrors: list of training errors
        lambdaRegularization: regularization parameter
        patience: number of epochs to wait before stopping

        returns True if the stopping criteria is met, False otherwise
        '''
        if patience == -1:
            if lambdaRegularization > 0:
                raise Warning("You are using regularization but not patience, the training will not stop")
            return False
        if lambdaRegularization > 0:
            if len(trainingErrors) < 10:
                return False
            #check if the training error is decreasing
            if trainingErrors[-1] < trainingErrors[-patience]*0.999:# improvements of at least 0.1%
                return False
            print(f'E now: {trainingErrors[-1]}')
            print(f'E before: {trainingErrors[-patience]}')
            print(f'percentage: {trainingErrors[-1]/trainingErrors[-patience]}')
            return True
        
        elif patience > 0:
            if len(validationErrors) < patience:
                return False
            #check if the validation error is increasing
            for i in range(len(validationErrors)-patience, len(validationErrors)):
                raise Warning("You are using early stopping but this is not supported yet for validation set")
                #print(validationErrors[i], validationErrors[i-1], validationErrors[i] < validationErrors[i-1])
                if validationErrors[i] < validationErrors[i-1]: #if it is improving, we stop even if the validation error remains the same
                    return False
            return True
        
        return False
    
    
    def variable_learning_rate(self, epoch:int, initial_learning_rate:float, tau:int) -> float:
        '''
        epoch: current epoch, which is tau in the formula
        initial_learning_rate: initial learning rate, which is eta_0 in the formula

        returns the new learning rate
        '''
        if tau==0:
            return initial_learning_rate

        eta_tau = 0.01*initial_learning_rate # 1% of the initial learning rate
        if epoch < tau:
            alpha = epoch/tau
            return (1-alpha)*initial_learning_rate + alpha*eta_tau # variable learning rate formula
        else:
            return eta_tau # after a certain number of epochs, we keep the learning rate equal to eta_tau
        

    def save_weights(self, path:str):
        '''
        path: path where to save the weights
        '''
        #create folder if it does not exist
        if not os.path.exists(path):
            os.makedirs(path)
        for i in range(len(self.layers)):
            np.savetxt(path+f"_{i}", self.layers[i].weights)
            np.savetxt(path+f"_{i}_bias", self.layers[i].bias)
    

    def load_weights(self, path:str):
        '''
        path: path where to load the weights
        '''
        for i in range(len(self.layers)):
            self.layers[i].weights = np.loadtxt(path+f"_{i}")
            self.layers[i].bias = np.loadtxt(path+f"_{i}_bias")

