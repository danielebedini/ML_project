import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))

import csv
import numpy as np
from net import *
from layers import *
from activations import *
from utilities import plot_data_error


def parse():
    #open file
    with open('./data/insurance.csv', newline='') as csvfile:
        #read file
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        #skip header
        next(reader)
        #read data in a loop
        data = np.array([])
        for row in reader:
            age = float(row[0])
            sex = float(row[1] == 'female')
            bmi = float(row[2])
            children = float(row[3])
            smoker = float(row[4] == 'yes')
            if row[5] == 'northeast':
                region = 0
            elif row[5] == 'northwest':
                region = 1
            elif row[5] == 'southeast':
                region = 2
            else:
                region = 3
            singleRow = [age, sex, bmi, children, smoker, region, float(row[6])]
            data = np.append(data, singleRow)
        data = data.reshape((len(data)//7, 7))
        #split data into 55% training, 15% validation and 30% test
        trainingData = data[:int(len(data)*0.55)]
        maxAge = np.max(trainingData[:,0])
        minAge = np.min(trainingData[:,0])
        avgAge = np.mean(trainingData[:,0])
        maxBmi = np.max(trainingData[:,2])
        minBmi = np.min(trainingData[:,2])
        avgBmi = np.mean(trainingData[:,2])
        maxChildren = np.max(trainingData[:,3])
        minChildren = np.min(trainingData[:,3])
        avgChildren = np.mean(trainingData[:,3])
        minCost = np.min(trainingData[:,6])
        maxCost = np.max(trainingData[:,6])
        #normalize data
        data[:,0] = (data[:,0] - avgAge)/(maxAge - minAge)
        data[:,2] = (data[:,2] - avgBmi)/(maxBmi - minBmi)
        data[:,3] = (data[:,3] - avgChildren)/(maxChildren - minChildren)
        data[:,6] = (data[:,6] - minCost)/(maxCost - minCost)
        trainingData = data[:int(len(data)*0.55)]
        validationData = data[int(len(data)*0.55):int(len(data)*0.7)]
        testData = data[int(len(data)*0.7):]
    return trainingData, validationData, testData

trainingData, validationData, testData = parse()

x_train = trainingData[:,:6]
y_train = trainingData[:,6]

x_val = validationData[:,:6]
y_val = validationData[:,6]

x_test = testData[:,:6]
y_test = testData[:,6]

insuranceNet = NeuralNet([LayerDense(6, 100, ActivationTanH()),
                          LayerDense(100, 50, ActivationTanH()),
                            LayerDense(50, 1, ActivationLinear())])


patience = []

exit()
trErrList, valErrList, _, _ = insuranceNet.train(x_train, y_train,
                    ValX=x_val, ValY=y_val,
                    epochs=2000,
                    r_prop=RProp(delta_0=0.01, delta_max=1),
                    patience=10,
                    lambdaRegularization=0.0001
                    )

plot_data_error(trErr, valErr, "training error", "validation error")
print("Test error: ", LossMSE(y_test, insuranceNet.forward(x_test)))
print("validation error: ", LossMSE(y_val, insuranceNet.forward(x_val)))

#print some predictions
print("Predictions: ")
print("Expected\tPredicted")
for i in range(10):
    print(y_test[i], "\t", insuranceNet.forward(x_test[i]))
