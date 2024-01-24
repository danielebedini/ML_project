import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))

import csv
import numpy as np
from net import *
from layers import *
from activations import *
from utilities import plot_data_error
from metrics import LossMSE, MEE

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
        global minCost
        minCost = np.min(trainingData[:,6])
        global maxCost 
        maxCost = np.max(trainingData[:,6])
        global avgCost
        avgCost = np.mean(trainingData[:,6])
        global stdVarCost 
        stdVarCost = np.std(trainingData[:,6])
        print("maxCost: ", maxCost)
        print("minCost: ", minCost)
        #normalize data
        data[:,0] = (data[:,0] - avgAge)/(maxAge - minAge)
        data[:,2] = (data[:,2] - avgBmi)/(maxBmi - minBmi)
        data[:,3] = (data[:,3] - avgChildren)/(maxChildren - minChildren)
        #data[:,6] = (data[:,6] - minCost)/(maxCost - minCost)
        data[:,6] = np.log(data[:,6])
        #data[:,6] = (data[:,6] - avgCost)/(stdVarCost)
        #data[:,6] = data[:,6]/10000
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


patience = [3, 5, 10, 15]
patience = lambda : np.random.randint(10, 13)
lambdas = [0.0002, 0.0005, 0.0007, 0.001, 0.002, 0.005, 0.007, 0.01, 0.02, 0.05, 0.07, 0.1]
lambdas = lambda : [0.001, 0.002, 0.003, 0.004, 0.005][np.random.randint(0, 5)]
delta_0 = [0.02, 0.05, 0.07, 0.1, 0.2, 0.5]
delta_0 = lambda : np.random.uniform(0.2, 0.4)
delta_max = [0.1, 0.2, 0.5, 1, 2, 5, 7, 10, 20]
delta_max = lambda : np.random.uniform(5, 20)
'''
patience = [3, 20]
lambdas = [0.0001, 0.1]
delta_0 = [0.01, 0.5]
delta_max = [0.1, 1]

'''
results = []

#print("tot iterations: ", len(patience)*len(lambdas)*len(delta_0)*len(delta_max))

#for p in patience:
#    for l in lambdas:
#        for d0 in delta_0:
#            for dm in delta_max:
for i in range(100):
    p = patience()
    l = lambdas()
    d0 = delta_0()
    dm = delta_max()

    _, _, _, _ = insuranceNet.train(x_train, y_train,
        epochs=500,
        r_prop=RProp(delta_0=d0, delta_max=dm),
        patience=p,
        lambdaRegularization=l
        )
    y_expected = np.exp(y_val)
    y_predicted = np.exp(insuranceNet.forward(x_val))
    mee = MEE(y_expected, y_predicted)
    results.append([p, l, d0, dm, mee])
    insuranceNet.reset()
    #print(int(len(results)/len(delta_max)/len(delta_0)/len(lambdas)/len(patience)*100), "%")
    print(i, "%")

results.sort(key=lambda x: x[4])

for r in results[0:20]:
    print(f'MEE: {"%.2f" % r[4]}\tPatience: {r[0]}\tDelta_0: {r[2]}\tDelta_max: {r[3]}\tLambda: {r[1]} ')

print("*********************"*5)

for r in results[-10:]:
    print(f'MEE: {"%.2f" % r[4]}\tPatience: {r[0]}\tDelta_0: {r[2]}\tDelta_max: {r[3]}\tLambda: {r[1]} ')

exit()
#save results
with open('results.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='|')
    for r in results:
        writer.writerow(r)

#plot_data_error(trErr, valErr, "training error", "validation error")
#y_expected = np.exp(y_test)
#y_predicted = np.exp(insuranceNet.forward(x_test))

#y_expected = y_test*(maxCost - minCost) + minCost
#y_predicted = insuranceNet.forward(x_test)*(maxCost - minCost) + minCost

#y_expected = y_test*stdVarCost + avgCost
#y_predicted = insuranceNet.forward(x_test)*stdVarCost + avgCost

y_expected = y_test*10000
y_predicted = insuranceNet.forward(x_test)*10000

print("Test MSE: ", LossMSE(y_expected, y_predicted))
print("Test MEE: ", MEE(y_expected, y_predicted))


#print some predictions
print("Predictions: ")
print("Expected\tPredicted")
for i in range(10):
    print(y_expected[i], "\t", y_predicted[i])
