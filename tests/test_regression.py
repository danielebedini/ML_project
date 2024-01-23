import sys
import os

sys.path.append(os.path.join(sys.path[0], '..'))
import tensorflow as tf
from net import NeuralNet
from layers import *
from activations import *
from metrics import LossMSE
from utilities import plot_data_error

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data()
# Reshape data for Boston Housing dataset
#x_train = x_train.reshape(404, 13)
#x_test = x_test.reshape(102, 13)

# Normalize data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_val = x_train[404-80:]
x_train = x_train[:404-80]

y_val = y_train[404-80:]
y_train = y_train[:404-80]



nn = NeuralNet([LayerDense(13, 100, ActivationTanH()),
                LayerDense(100, 1, ActivationLinear())])

robe = []
patiences = [3,5,10]
lambdas = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]

for i in patiences:
    for j in lambdas:
        trError, _, _, _ = nn.train(x_train, y_train, 
                                    lambdaRegularization=j,
                                    epochs=150, batch_size=-1, 
                                    r_prop=RProp(delta_0=0.01, delta_max=0.1),
                                    patience=i)
        valError = LossMSE(y_val, nn.forward(x_val))
        nn.reset()
        robe.append((trError[len(trError)-1], valError, i, j))
        print("Validation error: ",valError)

robe.sort(key=lambda x: x[1])

for i in robe:  
    print("Patience: ", i[2], "Lambda: ", i[3], "Training Error: ", i[0], "Validation Error: ", i[1])
    


# plot training and validation error
plot_data_error(trError, valError, firstName="Dio porco", secondName="Dio cane")

