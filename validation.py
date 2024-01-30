import numpy as np
from dataRescaling import DataProcessor
from learningRate import LearningRate
from net import NeuralNet
from utilities import plot_data_error

class Validator:

    def __init__(self, nn: NeuralNet, X:np.array, y:np.array, loss:callable, accuracy:callable= None, showPlot:bool=True):
        self.nn = nn
        self.X = X
        self.y = y
        self.loss = loss
        self.accuracy = accuracy
        self.showPlot = showPlot

        
    def kfold(self, k:int, epochs:int, learningRate:LearningRate = LearningRate(0.1), batch_size:int=-1, lambdaRegularization:float=0, momentum:float=0 ,patience:int=-1, r_prop:bool=False, outputProcessor:DataProcessor = None) -> (float, float, float, float):
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

        trainingError = []
        validationError = []
        trainingAccuracy = []
        validationAccuracy = []

        # shuffle data
        self.shuffleData()

        for fold in range(k):
            self.nn.reset()
            # create validation set
            valX = self.X[fold*valSize:(fold+1)*valSize]
            valy = self.y[fold*valSize:(fold+1)*valSize]
            # create training set
            trX = np.concatenate((self.X[:fold*valSize], self.X[(fold+1)*valSize:]))
            trY = np.concatenate((self.y[:fold*valSize], self.y[(fold+1)*valSize:]))

            #reshape data based only the training set
            if outputProcessor is not None:
                outputProcessor.reset(trY)
                trY = outputProcessor.process(trY)
                valy = outputProcessor.process(valy)

            # train
            trError, valError, trAccuracy, valAccuracy = self.nn.train(trX, trY, 
                                                ValX=valX, ValY=valy, # use given validation set
                                                learningRate=learningRate, 
                                                epochs=epochs, 
                                                batch_size=batch_size,
                                                momentum=momentum,
                                                lambdaRegularization=lambdaRegularization, 
                                                patience=patience, 
                                                r_prop=r_prop,
                                                accuracy=self.accuracy,
                                                printProgress=False
                                                )
            
            trE, valE, trA, valA = self.computeMetrics(outputProcessor, trX, trY, valX, valy)
            trainingErrorsList.append(trError)
            validationErrorsList.append(valError)
            trainingAccuracyList.append(trAccuracy)
            validationAccuracyList.append(valAccuracy)

            trainingError.append(trE)
            validationError.append(valE)
            if self.accuracy is not None:
                trainingAccuracy.append(trA)
                validationAccuracy.append(valA)
        
        if self.showPlot:
            self.kfoldPlot(trainingErrorsList, validationErrorsList, trainingAccuracyList, validationAccuracyList)

        # return the mean and variance of the metrics
        if self.accuracy is None:
            return np.mean(trainingError), np.mean(validationError), np.std(trainingError), np.std(valError), None, None, None, None
        else:
            return np.mean(trainingError), np.mean(validationError), np.std(trainingError), np.std(valError), np.mean(trainingAccuracy), np.mean(validationAccuracy), np.std(trainingAccuracy), np.std(validationAccuracy)
        

    def kfoldPlot(self, trLoss:[list], valLoss:[list], trAcc:[list], valAcc:[list]):
        # make all lists the same size
        maxSize = 0
        for i in trLoss:
            maxSize = max(len(i), maxSize)
        for i in range(len(trLoss)):
            while len(trLoss[i]) < maxSize:
                trLoss[i] = np.append(trLoss[i], trLoss[i][-1])
                valLoss[i] = np.append(valLoss[i], valLoss[i][-1])
                if self.accuracy is not None:
                    trAcc[i] = np.append(trAcc[i], trAcc[i][-1])
                    valAcc[i] = np.append(valAcc[i], valAcc[i][-1])

        meanTrLoss = np.mean(trLoss, axis=0)
        meanValLoss = np.mean(valLoss, axis=0)
        plot_data_error(meanTrLoss, meanValLoss, firstName="Tr_loss", secondName="Val_loss")

        if self.accuracy is not None:
            meanTrAcc = np.mean(trAcc, axis=0)
            meanValAcc = np.mean(valAcc, axis=0)
            plot_data_error(meanTrAcc, meanValAcc, firstName="Tr_acc", secondName="Val_acc")

    def computeMetrics(self, dataProcessor:DataProcessor, trX, trY, valX, valY) -> (float, float):

        y_predicted = self.nn.forward(trX)
        if dataProcessor is not None:
            y_expected = dataProcessor.deprocess(trY)
            y_predicted = dataProcessor.deprocess(y_predicted)
        else:
            y_expected = trY
        trError = self.loss(y_expected, y_predicted)
        if self.accuracy is not None:
            trAccuracy = self.accuracy(y_expected, y_predicted)
        else:
            trAccuracy = None
        
        y_predicted = self.nn.forward(valX)
        if dataProcessor is not None:
            y_expected = dataProcessor.deprocess(valY)
            y_predicted = dataProcessor.deprocess(y_predicted)
        else:
            y_expected = valY
        valError = self.loss(y_expected, y_predicted)
        if self.accuracy is not None:
            valAccuracy = self.accuracy(y_expected, y_predicted)
        else:
            valAccuracy = None

        return trError, valError, trAccuracy, valAccuracy

    def shuffleData(self):
        indexes = np.arange(len(self.X))
        np.random.shuffle(indexes)
        self.X = self.X[indexes]
        self.y = self.y[indexes]