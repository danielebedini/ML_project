import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))

from validation import Validator
from net import NeuralNet
from layers import *
from activations import *
from metrics import *
#from data.data3 import generate_data
from utilities import readMonkData, feature_one_hot_encoding, standard_one_hot_encoding, plot_data_error

# Create data
#X, y = generate_data(1000)

X, y = readMonkData("data/monk/monks-1.train")
X = feature_one_hot_encoding(X, [3,3,2,3,4,2])
y = standard_one_hot_encoding(y, 2)

nn = NeuralNet([LayerDense(17, 6, ActivationTanH()),
                LayerDense(6, 6, ActivationTanH()),
                LayerDense(6, 2, ActivationLinear())])

validator = Validator(nn, X, y, LossMSE, accuracy_classifier_multiple_output)

trainingErrors, validationErrors, trAccuracy, valAccuracy = validator.kfold(3, 300, 0.001, batch_size=50)


print("*********************")
print("Training Errors: ", trainingErrors)
print("Validation Errors: ", validationErrors)
print("*********************")
print("Training Accuracy: ", trAccuracy)
print("Validation Accuracy: ", valAccuracy)
print("*********************")
