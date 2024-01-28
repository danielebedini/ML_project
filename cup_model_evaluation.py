from threading import Thread
import time
from activations import *
from dataRescaling import DataProcessor
from grid_search import grid_search_RProp, grid_search_momentum, random_search_RProp
from layers import LayerDense
from learningRate import LearningRate
from metrics import MEE, LossMSE, rSquare
from net import NeuralNet
from r_prop_parameter import RProp
from utilities import plot_data_error, readTrainingCupData
from validation import Validator

X, y = readTrainingCupData("data/cup/ML-CUP23-TR.csv")

#take only 75% of the data for training and validation (the remaining 25% will be used for testing)
X = X[:int(len(X)*0.75)] #WARNING: do not touch this line
y = y[:int(len(y)*0.75)] #WARNING: do not touch this line


normalizer = DataProcessor(y, standardize=True, independentColumns=True)
y = normalizer.process(y)
valX = X[int(len(X)*0.75):]
valY = y[int(len(y)*0.75):]
X = X[:int(len(X)*0.75)]
y = y[:int(len(y)*0.75)]

nn_cup1 = NeuralNet([LayerDense(10, 30, ActivationTanH()),
                    LayerDense(30, 15, ActivationTanH()),
                    LayerDense(15, 3, ActivationLinear())
                    ],
                    name="tanH_30_15_3")

nn_cup1.reset(standardInit=False)
_, _, train, valid = nn_cup1.train(X, y, 
                             ValX = valX, ValY = valY, 
                             patience=9,
                             epochs=1500,
                             lambdaRegularization=0.00027,
                             accuracy=MEE,
                             r_prop=RProp(delta_0=0.09, delta_max=10), 
                             #accuracy=rSquare, 
                             printProgress=True)

plot_data_error(train, valid, "train", "val")

expected = normalizer.deprocess(y)
predicted = normalizer.deprocess(nn_cup1.forward(X))
print("tr  MEE: ", MEE(predicted, expected))
expected = normalizer.deprocess(valY)
predicted = normalizer.deprocess(nn_cup1.forward(valX))
print("val MEE: ", MEE(predicted, expected))

#show some predictions
for i in range(10):
    print("expected: ", expected[i], "predicted: ", predicted[i])
