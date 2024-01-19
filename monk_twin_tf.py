###############################################
################# tensorflow ##################
###############################################

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense  
from keras.optimizers import Adam
from utilities import feature_one_hot_encoding, readMonkData
from metrics import LossMSE, accuracy_classifier_single_output as accuracy

monk_num = 3

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
print("Training Accuracy: ", accuracy(y, y_predicted))
print("Training Loss: ", LossMSE(y, y_predicted))

#accuracy on test set
X, y = readMonkData(f"data/monk/monks-{monk_num}.test")
X = feature_one_hot_encoding(X, [3,3,2,3,4,2])
#y = standard_one_hot_encoding(y, 2)
y_predicted = model.predict(X)
print("Test Accuracy tf baby: ", accuracy(y, y_predicted))
print("Test Loss: ", LossMSE(y, y_predicted))