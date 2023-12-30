
from layers import *
from loss import LossMSE
from net import NeuralNet
from data3 import generate_data
# Create data
X, y = generate_data(1500)

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
                LayerDense(6, 2, ActivationLinear())])

nn.train(X, y, 0.01, 300, batch_size=-1)

#compute model error
y_predicted = nn.forward(X)
loss = LossMSE(y, y_predicted)
print("Loss: ", loss)
print(nn.forward([[1, 2]]))

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
print(model.predict([[1, 2]]))