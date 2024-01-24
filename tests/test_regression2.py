import sys
import os

sys.path.append(os.path.join(sys.path[0], '..'))
from net import NeuralNet
from layers import *
from activations import *
from metrics import LossMSE
from utilities import plot_data_error, feature_one_hot_encoding

# read data from insurance.csv

import csv

with open('insurance.csv', newline='') as csvfile:
    insurance = csv.reader(csvfile, delimiter=',', quotechar='|')
    data = []
    for row in insurance:
        data.append(row)

# remove first row
data = data[1:]

print(data[2])

x_train = np.array(data)
print(x_train[2])
y_train = np.array([])

for i in range(len(data)):
    for j in data[i]:
        if j=='female':
            x_train[i][1] = float(0)
        elif j=='male':
            x_train[i][1] = float(1)
        elif j=='northeast':
            x_train[i][5] = float(0)
        elif j=='northwest':
            x_train[i][5] = float(1)
        elif j=='southeast':
            x_train[i][5] = float(2)
        elif j=='southwest':
            x_train[i][5] = float(3)
        elif j=='yes':
            x_train[i][4] = float(1)
        elif j=='no':
            x_train[i][4] = float(0)
        elif j==data[i][len(data[i])-1]:
            y_train = np.append(y_train, float(data[i][len(data[i])-1]))

x_train = np.delete(x_train, len(x_train[0])-1, 1)

x_train = x_train.astype('float32')

# split data into train, validation and test set, 55% train, 15% validation, 30% test

x_test = x_train[int(len(x_train)*0.7):]
x_val = x_train[int(len(x_train)*0.55):int(len(x_train)*0.7)]
x_train = x_train[:int(len(x_train)*0.55)]

y_test = y_train[int(len(y_train)*0.7):]
y_val = y_train[int(len(y_train)*0.55):int(len(y_train)*0.7)]
y_train = y_train[:int(len(y_train)*0.55)]

# initialize neural network
nn = NeuralNet([LayerDense(6, 100, ActivationTanH()),
                LayerDense(100, 1, ActivationLinear())])

# rescale data betyween 0 and 1 with min-max normalization
x_train = (x_train - np.min(x_train)) / (np.max(x_train) - np.min(x_train))
y_train = (y_train - np.min(y_train)) / (np.max(y_train) - np.min(y_train))

# train neural network
trError, valError,_, _ = nn.train(x_train, y_train, 
                                  lambdaRegularization=0.0001,
                                  epochs=300, batch_size=-1, 
                                  r_prop=RProp(delta_0=0.01, delta_max=0.1),
                                  patience=10
                                  )

# plot training and validation error
plot_data_error(trError, valError, firstName="Training", secondName="Validation")
