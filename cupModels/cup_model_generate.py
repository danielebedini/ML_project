import sys
import os

sys.path.append(os.path.join(sys.path[0], '..'))

import numpy as np
from activations import *
from dataRescaling import DataProcessor
from layers import LayerDense
from metrics import MEE
from net import NeuralNet
from r_prop_parameter import RProp


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
