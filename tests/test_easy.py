import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))

from layers import *
from activations import *
from metrics import LossMSE
from net import NeuralNet
from data.data3 import generate_data
from utilities import plot_data_error
# Create data
X, y = generate_data(1000)
valX, valy = generate_data(100)

'''
layer1 = LayerDense(2, 5, ActivationReLU())
layer2 = LayerDense(5, 2, ActivationReLU())
layer3 = LayerDense(2, 1, ActivationLinear())

for epoch in range(1000):
    layer1.forward(X)
    layer2.forward(layer1.outputActivated)
    layer3.forward(layer2.outputActivated)
    loss = LossMSE(y, layer3.outputActivated)
    print("*************")
    #print(loss)
    #compute ouutput - expected output using numpy
    diff = np.subtract(layer3.outputActivated.T, y).T
    layer3.backward(diff)
    layer2.backward(layer3.delta, layer3.weights)
    layer1.backward(layer2.delta, layer2.weights)
'''

nn = NeuralNet([LayerDense(2, 6, ActivationTanH()),
                LayerDense(6, 6, ActivationTanH()),
                LayerDense(6, 2, ActivationLinear())])

trError, valError = nn.train(X, y, ValX=valX, ValY=valy, learningRate=0.05, epochs=10, batch_size=10, lambdaRegularization=0, patience=2, tau=40)

# plot training and validation error
plot_data_error(trError, valError)

#compute model error
y_predicted = nn.forward(X)
loss = LossMSE(y, y_predicted)
print("Loss: ", loss)

#predict some examples from the dataset
new_X, new_y = generate_data(100)
y_predicted = nn.forward(new_X)
print("Test Loss: ", LossMSE(new_y, y_predicted))

'''
for i in range(10):
    print("*************")
    print('new example: ', new_X[i])
    print('expected: ', new_y[i])
    print('real value: ', [np.sin(new_X[i][0]+new_X[i][1]), np.sin(new_X[i][0]-new_X[i][1])])
    print('predicted: ', nn.forward(new_X[i]))
'''
'''
#recreate the network with tensorflow
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Softmax

model = Sequential()
model.add(Dense(6, input_dim=2, activation='tanh'))
model.add(Dense(2, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=300, batch_size=149, verbose=0, validation_split=0)
#compute model error
y_predicted = model.predict(X)
loss = LossMSE(y, y_predicted)
print("Loss: ", loss)
print(model.predict([[0.9, 0.5]]))
'''