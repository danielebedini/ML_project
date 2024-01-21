import numpy as np
from layers import *
from activations import *
from net import NeuralNet
from utilities import feature_one_hot_encoding, readMonkData, standard_one_hot_encoding

# Here we can choose the monk dataset to use, number from 1 to 3
monk_num = 3
# Read the training data from the selected monk dataset
X, y = readMonkData(f"data/monk/monks-{monk_num}.train")

print(X.shape)
print(y.shape)

#one hot encode input
X = feature_one_hot_encoding(X, [3,3,2,3,4,2])
#y = standard_one_hot_encoding(y, 2)

monkModel = NeuralNet([LayerDense(17, 14, ActivationTanH()),
                        LayerDense(14, 10, ActivationTanH()),
                        LayerDense(10, 1, ActivationTanH())])

trError, valError = monkModel.train(X, y, epochs=600, batch_size=-1, lambdaRegularization=0.01, r_prop=RProp(delta_0=0.05))

#compute accuracy
from metrics import LossMSE, accuracy_classifier_single_output as accuracy
y_predicted = monkModel.forward(X)
print("Training Accuracy: ", accuracy(y, y_predicted))
#y_predicted = monkModel.forward(ValX)
#print("validation Accuracy: ", accuracy(ValY, y_predicted))
'''
#check test error
X, y = readMonkData(f"data/monk/monks-{monk_num}.test")
X = feature_one_hot_encoding(X, [3,3,2,3,4,2])
y = standard_one_hot_encoding(y, 2)
y_predicted = monkModel.forward(X)
print("Test Accuracy our baby: ", accuracy(y, y_predicted))
print("Test Loss: ", LossMSE(y, y_predicted))
'''
import matplotlib.pyplot as plt
plt.plot(trError, label="Training error")
plt.plot(valError, label="Validation error")
plt.legend()
plt.show()

#show some examples from the dataset
'''
start = 10
end = start + 10
new_X = X[start:end]
new_y = y[start:end]
y_predicted = monkModel.forward(new_X)
for i in range(len(new_X)):
    print("*************")
    print('new example: ', new_X[i])
    print('expected: ', new_y[i])
    print('predicted: ', y_predicted[i])
'''


