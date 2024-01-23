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
        trainingErrorsList = []
        validationErrorsList = []
        trainingAccuracyList = []
        validationAccuracyList = []

        trainingError = 0
        validationError = 0
        trainingAccuracy = 0
        validationAccuracy = 0


        for fold in range(k):
            # create validation set
            valX = self.X[fold*valSize:(fold+1)*valSize]
            valy = self.y[fold*valSize:(fold+1)*valSize]

            # create training set
            trX = np.concatenate((self.X[:fold*valSize], self.X[(fold+1)*valSize:]))
            trY = np.concatenate((self.y[:fold*valSize], self.y[(fold+1)*valSize:]))

            # train
            trError, valError, trAccuracy, valAccuracy = self.nn.train(trX, trY, 
                                                ValX=valX, ValY=valy, # use given validation set
                                                learningRate=learningRate, 
                                                epochs=epochs, 
                                                batch_size=batch_size,
                                                momentum=momentum,
                                                lambdaRegularization=lambdaRegularization, 
                                                patience=patience, 
                                                tau=tau,
                                                r_prop=r_prop,
                                                accuracy=self.accuracy)
            trainingErrorsList.append(trError)
            validationErrorsList.append(valError)
            trainingAccuracyList.append(trAccuracy)
            validationAccuracyList.append(valAccuracy)

            trainingError += trError[-1]/k
            validationError += valError[-1]/k
            if trAccuracy is not None and valAccuracy is not None:
                trainingAccuracy += trAccuracy[-1]/k
                validationAccuracy += valAccuracy[-1]/k
            # reset the network
            self.nn.reset()
        
        self.kfoldPlot(trainingErrorsList, validationErrorsList, trainingAccuracyList, validationAccuracyList)

        # return the mean of the metrics

        if trainingAccuracy == 0 and validationAccuracy == 0:
            return np.mean(trainingError), np.mean(validationError), None, None
        else:
            return np.mean(trainingError), np.mean(validationError), np.mean(trainingAccuracy), np.mean(validationAccuracy)
        

    def kfoldPlot(self, trLoss:[list], valLoss:[list], trAcc:[list], valAcc:[list]):
        # make all lists the same size
        maxSize = 0
        for i in trLoss:
            maxSize = max(len(i), maxSize)
        for i in range(len(trLoss)):
            while len(trLoss[i]) < maxSize:
                trLoss[i] = np.append(trLoss[i], trLoss[i][-1])
                valLoss[i] = np.append(valLoss[i], valLoss[i][-1])
                if trAcc is not None and valAcc is not None:
                    trAcc[i] = np.append(trAcc[i], trAcc[i][-1])
                    valAcc[i] = np.append(valAcc[i], valAcc[i][-1])

        meanTrLoss = np.mean(trLoss, axis=0)
        meanValLoss = np.mean(valLoss, axis=0)
        plot_data_error(meanTrLoss, meanValLoss, firstName="Tr_loss", secondName="Val_loss")

        if trAcc is not None and valAcc is not None:
            meanTrAcc = np.mean(trAcc, axis=0)
            meanValAcc = np.mean(valAcc, axis=0)
            plot_data_error(meanTrAcc, meanValAcc, firstName="Tr_acc", secondName="Val_acc")


