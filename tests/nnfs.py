import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))

from data.data import createData
from net import NeuralNet
from layers import *
from activations import *
from utilities import plot_data_error, standard_one_hot_encoding
from metrics import LossMSE, accuracy_classifier_multiple_output as accuracy

X, y = createData(500, 2)
print(X.shape)
print(y.shape)

#randomly shuffle the data
p = np.random.permutation(len(X))
X = X[p]
y = y[p]

#normalize data
X = X - np.mean(X, axis=0)
X = X / np.std(X, axis=0)
y = standard_one_hot_encoding(y, 2)

nn = NeuralNet([LayerDense(2, 10, ActivationTanH()),
                LayerDense(10, 12, ActivationTanH()),
                LayerDense(12, 16, ActivationTanH()),
                LayerDense(16, 16, ActivationTanH()),
                LayerDense(16, 12, ActivationTanH()),
                LayerDense(12, 5, ActivationTanH()),
                LayerDense(5, 2, ActivationTanH())])


#check initia accuracy
y_predicted = nn.forward(X)
print("Initial Accuracy: ", accuracy(y, y_predicted))

trError, valError = nn.train(X, y, epochs=500, batch_size=-1, r_prop=RProp(delta_0=0.01, delta_max=0.1))

#compute accuracy
y_predicted = nn.forward(X)
print("training Accuracy: ", accuracy(y, y_predicted))
plot_data_error(trError, valError)


###############################################
################# tensorflow ##################
###############################################

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim=2, activation='tanh'))
model.add(Dense(12, activation='tanh'))
model.add(Dense(16, activation='tanh'))
model.add(Dense(16, activation='tanh'))
model.add(Dense(12, activation='tanh'))
model.add(Dense(5, activation='tanh'))
model.add(Dense(2, activation='tanh'))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

#accuracy before training
y_predicted = model.predict(X)
print("Initial Accuracy: ", accuracy(y, y_predicted))

hystory = model.fit(X, y, epochs=250, batch_size=None)

#accuracy after training
y_predicted = model.predict(X)
print("training Accuracy: ", accuracy(y, y_predicted))
print("training Loss: ", LossMSE(y, y_predicted))
#plot training and validation error
lossTF = hystory.history['loss']
plot_data_error(lossTF, lossTF)


