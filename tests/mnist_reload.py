import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))

from net import NeuralNet
from layers import *
from activations import *
from metrics import LossMSE
from keras.datasets import mnist
from utilities import plot_data_error, standard_one_hot_encoding

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape data
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

# Normalize data
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

y_train = y_train.astype('int32')
y_test = y_test.astype('int32')
y_train = standard_one_hot_encoding(y_train, 10)
y_test = standard_one_hot_encoding(y_test, 10)

X_val = X_train[50000:]
y_val = y_train[50000:]
X_train = X_train[:50000]
y_train = y_train[:50000]

nn = NeuralNet([LayerDense(784, 100, ActivationTanH()),
                LayerDense(100, 10, ActivationSoftmax())])

#load network weights
nn.load_weights("weights/mnist_weights")

#compute accuracy on training set
from metrics import accuracy_classifier_multiple_output as accuracy
y_predicted = nn.forward(X_train)
print("Training Accuracy: ", accuracy(y_train, y_predicted))
print("Training Loss: ", LossMSE(y_train, y_predicted))

#compute accuracy on validation set
y_predicted = nn.forward(X_val)
print("Validation Accuracy: ", accuracy(y_val, y_predicted))
print("Validation Loss: ", LossMSE(y_val, y_predicted))

# compute accuracy  
y_predicted = nn.forward(X_test)
print("Test Accuracy: ", accuracy(y_test, y_predicted))
print("Test Loss: ", LossMSE(y_test, y_predicted))

#show some examples from the dataset

import matplotlib.pyplot as plt
start = 10
end = start + 10
new_X = X_test[start:end]
new_y = y_test[start:end]
y_predicted = nn.forward(new_X)
for i in range(len(new_X)):
    print("*************")
    #show exaple in pyplot
    expected = np.argmax(new_y[i]) + 1
    if expected == 10: expected = 0
    print('expected: ', expected)
    predicted = np.argmax(y_predicted[i]) + 1
    if predicted == 10: predicted = 0
    print('predicted: ', predicted)
    plt.imshow(new_X[i].reshape(28,28), cmap='gray')
    plt.show()