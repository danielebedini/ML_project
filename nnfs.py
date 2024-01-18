from data.data import createData
from net import NeuralNet
from layers import *
from utilities import plot_data_error, standard_one_hot_encoding

X, y = createData(200, 2)
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

nn = NeuralNet([LayerDense(2, 10, ActivationLeakyReLU()),
                LayerDense(10, 12, ActivationLeakyReLU()),
                LayerDense(12, 16, ActivationTanH()),
                LayerDense(16, 16, ActivationTanH()),
                LayerDense(16, 12, ActivationTanH()),
                LayerDense(12, 5, ActivationTanH()),
                LayerDense(5, 2, ActivationTanH())])


#check initia accuracy
y_predicted = nn.forward(X)
from metrics import accuracy_classifier_multiple_output as accuracy
print("Initial Accuracy: ", accuracy(y, y_predicted))

trError, valError = nn.train(X, y, learningRate=0.002, epochs=1000, batch_size=20)

#compute accuracy
y_predicted = nn.forward(X)
print("training Accuracy: ", accuracy(y, y_predicted))
plot_data_error(trError, valError)

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim=2, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(16, activation='tanh'))
model.add(Dense(16, activation='tanh'))
model.add(Dense(12, activation='tanh'))
model.add(Dense(5, activation='tanh'))
model.add(Dense(2, activation='tanh'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

hystory = model.fit(X, y, epochs=200, batch_size=20)
#plot training and validation error
lossTF = hystory.history['loss']
accuracyTF = hystory.history['accuracy']
plot_data_error(lossTF, accuracyTF)


