import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))

import time
from activations import *
from dataRescaling import DataProcessor
from layers import LayerDense
from metrics import MEE
from net import NeuralNet
from r_prop_parameter import RProp
from utilities import plot_data_error, readTrainingCupData

X, y = readTrainingCupData("data/cup/ML-CUP23-TR.csv")

#take only 75% of the data for training and validation (the remaining 25% will be used for testing)
X = X[:int(len(X)*0.75)] #WARNING: do not touch this line
y = y[:int(len(y)*0.75)] #WARNING: do not touch this line

testX = X[int(len(X)*0.75):]
testY = y[int(len(y)*0.75):]


def generateGoodModel(trainingData:np.ndarray, trainingLabels:np.ndarray) -> NeuralNet:

    trainingLabelsCopy = trainingLabels.copy()
    standardizer = DataProcessor(trainingLabelsCopy, standardize=True, independentColumns=True)
    trainingLabelsCopy = standardizer.process(trainingLabelsCopy)

    nn = NeuralNet([LayerDense(10, 30, ActivationTanH(True)),
                    LayerDense(30, 15, ActivationTanH(True)),
                    LayerDense(15, 3, ActivationLinear(True))
                    ],
                    name="tanH_30_15_3")
    while True:
        nn.reset()
        _, _, trainingErrors, _ = nn.train(trainingData, trainingLabelsCopy,
                                            epochs=4000,
                                            lambdaRegularization=0.000013,
                                            patience=10,
                                            accuracy=MEE,
                                            r_prop=RProp(delta_0=0.07, delta_max=50),
                                            )

        trainingError = MEE(trainingLabels, standardizer.deprocess(nn.forward(trainingData)))
        if trainingError < 0.65: #avoid considering underfitting models
            if len(trainingErrors) < 4000: #if the model did not converge, reject it
                return nn
        print("rejected model, training error: ", trainingError, "training epoch: ", len(trainingErrors))

class Ensemble:
    def __init__(self, trainingData:np.ndarray, trainingLabels:np.ndarray, ensembleSize:int):
        self.outputStandardizer = DataProcessor(trainingLabels, standardize=True, independentColumns=True)
        self.models = []
        self.createEnsemble(trainingData, trainingLabels, ensembleSize)

    def createEnsemble(self, trainingData:np.ndarray, trainingLabels:np.ndarray, ensembleSize:int) -> list:
        for i in range(ensembleSize):
            self.models.append(generateGoodModel(trainingData, trainingLabels))
    
    def predict(self, input:np.ndarray) -> np.ndarray:
        predictions = []
        for model in self.models:
            predictions.append(model.forward(input))
        predictions = np.array(predictions)
        predictions = np.mean(predictions, axis=0)
        predictions = self.outputStandardizer.deprocess(predictions)
        return predictions

start = time.time()

ensembles = []
for i in range(10):
    ensembles.append(Ensemble(X, y, 10))

end = time.time()
print("Training time: ", end-start)

testErrors = []
for ensemble in ensembles:
    testErrors.append(MEE(testY, ensemble.predict(testX)))

print("Test errors: ", testErrors)
print("Average test error: ", np.mean(testErrors))
print("Test error standard deviation: ", np.std(testErrors))


exit()
#show some predictions
for i in range(10):
    print("expected: ", expected[i], "predicted: ", predicted[i])
