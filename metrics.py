import numpy as np


def LossMSE(expectedOutputs, outputs):
    '''
    this function calculates the mean squared error loss
    expectedOutputs: the expected outputs of the activation function
    outputs: the outputs of the activation function
    '''
    if outputs.shape[1] == 1: outputs = np.reshape(outputs, outputs.shape[0])
    return np.sum((expectedOutputs - outputs) ** 2) / expectedOutputs.size


def MEE(expectedOutputs, outputs):
    '''
    this function calculates the mean euclidean error loss
    expectedOutputs: the expected outputs of the activation function
    outputs: the outputs of the activation function
    '''
    if outputs.shape[1] == 1: outputs = np.reshape(outputs, outputs.shape[0])
    return np.sum(np.sqrt(np.sum((expectedOutputs - outputs) ** 2))) / expectedOutputs.shape[0]


def accuracy_classifier_single_output(expectedOutputs, outputs):
    '''
    this function calculates the accuracy of the model
    expectedOutputs: the expected outputs of the activation function
    outputs: the outputs of the activation function

    note: outputs are assumed to be in the range [0,1] and just one output
    '''
    correct = 0
    for i in range(len(expectedOutputs)):
        if expectedOutputs[i] == (outputs[i] > 0.5):
            correct += 1
    return correct / len(expectedOutputs)


def accuracy_classifier_multiple_output(expectedOutputs, outputs):
    '''
    this function calculates the accuracy of the model
    expectedOutputs: the expected outputs of the activation function
    outputs: the outputs of the activation function

    note: outputs are assumed to be a vector of probabilities (one hot encoding)
    '''
    correct = 0
    for i in range(len(expectedOutputs)):
        if np.argmax(expectedOutputs[i]) == np.argmax(outputs[i]):
            correct += 1
    return correct / len(expectedOutputs)

def rSquare(expectedOutputs, outputs):
    '''
    this function calculates the R^2 coefficient of determination
    expectedOutputs: the expected outputs of the activation function
    outputs: the outputs of the activation function
    '''
    if outputs.shape[1] == 1: outputs = np.reshape(outputs, outputs.shape[0])
    return 1 - np.sum((expectedOutputs - outputs) ** 2) / np.sum((expectedOutputs - np.mean(expectedOutputs)) ** 2)