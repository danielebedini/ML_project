import numpy as np


def LossMSE(expectedOutputs, outputs):
    '''
    this function calculates the mean squared error loss
    expectedOutputs: the expected outputs of the activation function
    outputs: the outputs of the activation function
    '''
    return np.sum((expectedOutputs - outputs) ** 2) / len(expectedOutputs)