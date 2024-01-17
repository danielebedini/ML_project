import numpy as np


def LossMSE(expectedOutputs, outputs):
    '''
    this function calculates the mean squared error loss
    expectedOutputs: the expected outputs of the activation function
    outputs: the outputs of the activation function
    '''
    return np.sum((expectedOutputs - outputs) ** 2) / expectedOutputs.size

def accuracy_classifier(expectedOutputs, outputs):
    '''
    this function calculates the accuracy of the model
    expectedOutputs: the expected outputs of the activation function
    outputs: the outputs of the activation function
    '''
    correct = 0
    for i in range(len(expectedOutputs)):
        if expectedOutputs[i] == (outputs[i] > 0.5):
            correct += 1
    return correct / len(expectedOutputs)