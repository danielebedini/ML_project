from net import NeuralNet
from layers import *
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

nn = NeuralNet([LayerDense(784, 100, ActivationLeakyReLU()),
                LayerDense(100, 10, ActivationSoftmax())])

trError, valError = nn.train(X_train, y_train, ValX=X_test, ValY=y_test, learningRate=0.05, epochs=300, batch_size=10, lambdaRegularization=0, patience=-1, tau=100)

# plot training and validation error
plot_data_error(trError, valError)

