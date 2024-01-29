import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))

import numpy as np
from layers import *
from activations import *
from net import NeuralNet
from utilities import feature_one_hot_encoding, readMonkData, standard_one_hot_encoding
from metrics import LossMSE, accuracy_classifier_single_output as accuracy, MEE

# Here we can choose the monk dataset to use, number from 1 to 3
monk_num = 3
# Read the training data from the selected monk dataset
X, y = readMonkData(f"data/monk/monks-{monk_num}.train")

print(X.shape)
print(y.shape)

#one hot encode input
X = feature_one_hot_encoding(X, [3,3,2,3,4,2])
#y = standard_one_hot_encoding(y, 2)

monkModel = NeuralNet([LayerDense(17, 4, ActivationTanH()),
                        LayerDense(4, 1, ActivationTanH())])

trError, valError, trAccuracy, valAccuracy = monkModel.train(X, y, epochs=100, r_prop=RProp(), lambdaRegularization=0.07, patience=3, accuracy=accuracy)

#compute accuracy
y_predicted = monkModel.forward(X)
print("Training Accuracy: ", accuracy(y, y_predicted))
print("Training MSE: ", LossMSE(y, y_predicted))
print("Training MEE: ", MEE(y, y_predicted))

#check test error
X, y = readMonkData(f"data/monk/monks-{monk_num}.test")
X = feature_one_hot_encoding(X, [3,3,2,3,4,2])
y_predicted = monkModel.forward(X)
print("Test Accuracy our baby: ", accuracy(y, y_predicted))
print("Test MSE: ", LossMSE(y, y_predicted))
print("Test MEE: ", MEE(y, y_predicted))

import matplotlib.pyplot as plt
plt.plot(trError, label="Training error")
plt.legend()
plt.show()

plt.plot(trAccuracy, label="Training accuracy")
plt.legend()
plt.show()