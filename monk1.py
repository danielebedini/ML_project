import numpy as np
from layers import ActivationLinear, ActivationReLU, ActivationTanH, LayerDense
from net import NeuralNet
from utilities import feature_one_hot_encoding, readMonkData, standard_one_hot_encoding

# Path: monk1.py
# Read the training data
X, y = readMonkData("data/monk/monks-1.train")

print(X.shape)
print(y.shape)

#one hot encode input
X = feature_one_hot_encoding(X, [3,3,2,3,4,2])
y = standard_one_hot_encoding(y, 2)

#split validation set giving 30% of the data to validation
'''totData = X.shape[0]
trainData = int(totData * 0.7)
ValX = X[trainData:]
ValY = y[trainData:]
X = X[:trainData]
y = y[:trainData]
print(X.shape)
print(y.shape)'''

monkModel = NeuralNet([LayerDense(17, 10, ActivationTanH()),
                        LayerDense(10, 2, ActivationLinear())])

#trError, valError = monkModel.train(X, y, ValX, ValY, learningRate=0.002, epochs=100, batch_size=30)
trError, valError = monkModel.train(X, y, learningRate=0.003, epochs=300, batch_size=30, lambdaRegularization=0.0001)

#compute accuracy
from metrics import LossMSE, accuracy_classifier_multiple_output as accuracy
y_predicted = monkModel.forward(X)
print("training Accuracy: ", accuracy(y, y_predicted))
#y_predicted = monkModel.forward(ValX)
#print("validation Accuracy: ", accuracy(ValY, y_predicted))

#check test error
'''X, y = readMonkData("data/monk/monks-1.test")
X = feature_one_hot_encoding(X, [3,3,2,3,4,2])
y = standard_one_hot_encoding(y, 2)
y_predicted = monkModel.forward(X)
print("Test Accuracy: ", accuracy(y, y_predicted))
print("Test Loss: ", LossMSE(y, y_predicted))   '''

import matplotlib.pyplot as plt
plt.plot(trError, label="Training error")
plt.plot(valError, label="Validation error")
plt.legend()
plt.show()

#show some examples from the dataset
'''start = 10
end = start + 10
new_X = X[start:end]
new_y = y[start:end]
y_predicted = monkModel.forward(new_X)
for i in range(len(new_X)):
    print("*************")
    print('new example: ', new_X[i])
    print('expected: ', new_y[i])
    print('predicted: ', y_predicted[i])'''

positive = 0
negative = 0
for i in range(len(X)):
    if np.argmax(y[i]) == 1:
        positive += 1
    else:
        negative += 1

#print("positive: ", positive)
#print("negative: ", negative)