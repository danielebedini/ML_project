import numpy as np
from layers import ActivationLinear, ActivationReLU, ActivationTanH, LayerDense
from net import NeuralNet
from utils import feature_one_hot_encoding, readMonkData, standard_one_hot_encoding

# Path: monk1.py
# Read the training data
X, y = readMonkData("data/monk/monks-1.train")

print(X.shape)
print(y.shape)

#one hot encode input
X = feature_one_hot_encoding(X, [3,3,2,3,4,2])
#y = standard_one_hot_encoding(y, 2)

#split validation set giving 30% of the data to validation
totData = X.shape[0]
trainData = int(totData * 0.99)
print(X.shape)
print(y.shape)

monkModel = NeuralNet([LayerDense(17, 4, ActivationTanH()),
                          LayerDense(4, 1, ActivationTanH())])

trError, valError = monkModel.train(X, y, learningRate=0.004, epochs=2500, batch_size=-1)
#[5, 2, 1, 1, 3, 2] -> [[0,0,0,0,1], [0,1,0], [1,0], [1,0], [0,0,1], [0,1]]

#compute accuracy
from metrics import accuracy_classifier as accuracy
y_predicted = monkModel.forward(X)
print("training Accuracy: ", accuracy(y, y_predicted))
#y_predicted = monkModel.forward(ValX)
#print("validation Accuracy: ", accuracy(ValY, y_predicted))


import matplotlib.pyplot as plt
plt.plot(trError, label="Training error")
plt.plot(valError, label="Validation error")
plt.legend()
plt.show()

#show some examples from the dataset
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

# count positive examples in the training set
count = 0
for i in range(len(y)):
    if y[i][0] == 1:
        count += 1
print("positive examples in the training set: ", count)
print("negative examples in the training set: ", len(y) - count)