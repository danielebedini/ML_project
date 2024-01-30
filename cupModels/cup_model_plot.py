import sys
import os

sys.path.append(os.path.join(sys.path[0], '..'))

import time
from activations import *
from dataRescaling import DataProcessor
from layers import LayerDense
from validation import Validator
from metrics import MEE
from net import NeuralNet
from r_prop_parameter import RProp
from utilities import plot_data_error, readTrainingCupData

X, y = readTrainingCupData("data/cup/ML-CUP23-TR.csv")

#take only 75% of the data for training and validation (the remaining 25% will be used for testing)
X = X[:int(len(X)*0.75)] #WARNING: do not touch this line
y = y[:int(len(y)*0.75)] #WARNING: do not touch this line


nn = NeuralNet([LayerDense(10, 30, ActivationTanH(True)),
                LayerDense(30, 15, ActivationTanH(True)),
                LayerDense(15, 3, ActivationLinear(True))
                ],
                name="tanH_30_15_3")

validator = Validator(nn, X, y, MEE)

trainingErrors, validationErrors, _, _, _, _, _, _ = validator.kfold(k=10, 
                                                                        epochs=4000, 
                                                                        r_prop=RProp(delta_0=0.07, delta_max=50), 
                                                                        lambdaRegularization=0.000013, 
                                                                        patience=10,
                                                                        outputProcessor=DataProcessor(y, standardize=True, independentColumns=True)
                                                                     )