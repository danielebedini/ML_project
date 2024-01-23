import numpy as np
from net import NeuralNet
from utilities import plot_data_error

class Validator:

    def __init__(self, nn: NeuralNet, X:np.array, y:np.array, loss:callable, accuracy:callable= None ):
        self.nn = nn
        self.X = X
        self.y = y
        self.loss = loss
        self.accuracy = accuracy
        #self.y_predicted = None
        #self.accuracy = None

        
    def kfold(self, k:int, epochs:int, learningRate:float = 0.5, batch_size:int=-1, lambdaRegularization:float=0, momentum:float=0 ,patience:int=-1, tau:int=0, r_prop:bool=False):
        '''
        k: number of folds
        epochs: number of epochs
        learningRate: learning rate
        batch_size: batch size
        '''
        valSize = int(len(self.X) / k) # size of each fold
        trainingErrors = np.array([])
        validationErrors = np.array([])
        trainingAccuracy = np.array([])
        validationAccuracy = np.array([])

        for fold in range(k):
            # create validation set
            valX = self.X[fold*valSize:(fold+1)*valSize]
            valy = self.y[fold*valSize:(fold+1)*valSize]

            # create training set
            trX = np.concatenate((self.X[:fold*valSize], self.X[(fold+1)*valSize:]))
            trY = np.concatenate((self.y[:fold*valSize], self.y[(fold+1)*valSize:]))

            # train
            trError, valError = self.nn.train(trX, trY, 
                                                ValX=valX, ValY=valy, # use given validation set
                                                learningRate=learningRate, 
                                                epochs=epochs, 
                                                batch_size=batch_size,
                                                momentum=momentum,
                                                lambdaRegularization=lambdaRegularization, 
                                                patience=patience, 
                                                tau=tau,
                                                r_prop=r_prop)
            # calculate loss and accuracy in according to the chosen metrics
            # loss on training set
            y_predicted = self.nn.forward(trX)
            loss = self.loss(trY, y_predicted)
            trainingErrors = np.append(trainingErrors, loss)

            # accuracy on training set
            if self.accuracy is not None:
                acc = self.accuracy(trY, y_predicted)
                trainingAccuracy = np.append(trainingAccuracy, acc)

            # loss on validation set
            y_predicted = self.nn.forward(valX)
            loss = self.loss(valy, y_predicted)
            validationErrors = np.append(validationErrors, loss)
            
            # accuracy on validation set
            if self.accuracy is not None:
                acc = self.accuracy(valy, y_predicted)
                validationAccuracy = np.append(validationAccuracy, acc)
            
            # reset the network
            self.nn.reset()
        
        # return the mean of the metrics
        if trainingAccuracy.size == 0 and validationAccuracy.size == 0:
            return np.mean(trainingErrors), np.mean(validationErrors), None, None
        else:
            return np.mean(trainingErrors), np.mean(validationErrors), np.mean(trainingAccuracy), np.mean(validationAccuracy)


