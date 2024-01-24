from activations import *
from layers import LayerDense
from learningRate import LearningRate
from net import NeuralNet
from r_prop_parameter import RProp
from utilities import plot_data_error, readTrainingCupData

X, y = readTrainingCupData("data/cup/ML-CUP23-TR.csv")

print(X.shape)
print(y.shape)

#take only 50% of test data
X = X[:int(len(X)*0.5)] #WARNING: do not touch this line
y = y[:int(len(y)*0.5)] #WARNING: do not touch this line


#####################################################################
# during experimentig phase apply changese onlu from here and below #
# | |                     |         |                           | | #
# V V                     V         V                           V V #
#####################################################################

nn_cup = NeuralNet([LayerDense(10, 20, ActivationTanH()),
                    LayerDense(20, 10, ActivationTanH()),
                    LayerDense(10, 3, ActivationLinear())
                    ])

trErr, valErr, _, _ = nn_cup.train(X, y, epochs=200, 
            learningRate=LearningRate(0.05),
            batch_size=100,
            lambdaRegularization=0.0,
            momentum=0.9,
            patience=-1)


plot_data_error(trErr, valErr, "training error", "validation error")