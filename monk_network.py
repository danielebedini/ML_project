import numpy as np
from layers import *
from net import NeuralNet
from utilities import feature_one_hot_encoding, readMonkData, standard_one_hot_encoding

monk_num = 3
# Path: monk1.py
# Read the training data
X, y = readMonkData(f"data/monk/monks-{monk_num}.train")

print(X.shape)
print(y.shape)

#one hot encode input
X = feature_one_hot_encoding(X, [3,3,2,3,4,2])
#y = standard_one_hot_encoding(y, 2)

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
                        LayerDense(10, 10, ActivationTanH()),
                        LayerDense(10, 1, ActivationTanH())])

trError, valError = monkModel.train(X, y, learningRate=0.001, epochs=200, batch_size=30, lambdaRegularization=0, momentum=0.99)

#compute accuracy
from metrics import LossMSE, accuracy_classifier_single_output as accuracy
y_predicted = monkModel.forward(X)
print("training Accuracy: ", accuracy(y, y_predicted))
#y_predicted = monkModel.forward(ValX)
#print("validation Accuracy: ", accuracy(ValY, y_predicted))

#check test error
X, y = readMonkData(f"data/monk/monks-{monk_num}.test")
X = feature_one_hot_encoding(X, [3,3,2,3,4,2])
#y = standard_one_hot_encoding(y, 2)
y_predicted = monkModel.forward(X)
print("Test Accuracy our baby: ", accuracy(y, y_predicted))
print("Test Loss: ", LossMSE(y, y_predicted))

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

'''positive = 0
negative = 0
for i in range(len(X)):
    if np.argmax(y[i]) == 1:
        positive += 1
    else:
        negative += 1

print("positive: ", positive)
print("negative: ", negative)'''

###############################################
################# tensorflow ##################
###############################################

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense  
from keras.optimizers import Adam

optimizerAdam = Adam(learning_rate=0.001, beta_1=0.99, beta_2=0, amsgrad=False)

model = Sequential()
model.add(Dense(10, input_dim=17, activation='tanh'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(1, activation='tanh'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

X, y = readMonkData(f"data/monk/monks-{monk_num}.train")
X = feature_one_hot_encoding(X, [3,3,2,3,4,2])
#accuracy before training
y_predicted = model.predict(X)
print("Initial Accuracy: ", accuracy(y, y_predicted))

hystory = model.fit(X, y, epochs=100, batch_size=30, verbose=0)

#accuracy after training
y_predicted = model.predict(X)
print("training Accuracy: ", accuracy(y, y_predicted))
print("training Loss: ", LossMSE(y, y_predicted))

#accuracy on test set
X, y = readMonkData(f"data/monk/monks-{monk_num}.test")
X = feature_one_hot_encoding(X, [3,3,2,3,4,2])
#y = standard_one_hot_encoding(y, 2)
y_predicted = model.predict(X)
print("Test Accuracy tf baby: ", accuracy(y, y_predicted))
print("Test Loss: ", LossMSE(y, y_predicted))