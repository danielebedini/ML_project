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

# Here we can choose the monk dataset to use, number from 1 to 3
monk_num = 2
# Read the training data from the selected monk dataset
X, y = readMonkData(f"data/monk/monks-{monk_num}.train")

print(X.shape)
print(y.shape)

#one hot encode input
X = feature_one_hot_encoding(X, [3,3,2,3,4,2])
#y = standard_one_hot_encoding(y, 2)

nn = NeuralNet([LayerDense(17, 4, ActivationTanH()),
                LayerDense(4, 1, ActivationTanH())])

validator = Validator(nn, X, y, LossMSE, accuracy_classifier_single_output)

trainingErrors, validationErrors, trAccuracy, valAccuracy = validator.kfold(k=15, epochs=250, learningRate=0.5, batch_size=-1, lambdaRegularization=0.0005)


print("*********************")
print("Training Errors: ", trainingErrors)
print("Validation Errors: ", validationErrors)
print("*********************")
print("Training Accuracy: ", trAccuracy)
print("Validation Accuracy: ", valAccuracy)
print("*********************")
