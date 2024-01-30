import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))


###############################################
################# tensorflow ##################
###############################################

#NOTE: this is a file that we used while imlementing our network to see if we 
# were on the right track. It is not used in the final report.
from keras.models import Sequential
from keras.layers import Dense  
from keras.optimizers import Adam
from utilities import feature_one_hot_encoding, readMonkData, plot_data_error
from metrics import LossMSE, accuracy_classifier_single_output as accuracy

monk_num = 1

optimizerAdam = Adam(learning_rate=0.05, beta_1=0.99, beta_2=0, amsgrad=False)

model = Sequential()
model.add(Dense(4, input_dim=17, activation='tanh'))
#model.add(Dense(10, activation='tanh'))
model.add(Dense(1, input_dim=4, activation='tanh'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

X, y = readMonkData(f"data/monk/monks-{monk_num}.train")
X = feature_one_hot_encoding(X, [3,3,2,3,4,2])
#accuracy before training
y_predicted = model.predict(X)
print("Initial Accuracy: ", accuracy(y, y_predicted))

hystory = model.fit(X, y, epochs=60, batch_size=10, verbose=0)

plot_data_error(hystory.history['loss'], [], "Loss", "")

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