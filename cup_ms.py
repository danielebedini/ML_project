from activations import *
from dataRescaling import DataProcessor
from layers import LayerDense
from learningRate import LearningRate
from metrics import MEE, LossMSE
from net import NeuralNet
from r_prop_parameter import RProp
from utilities import plot_data_error, readTrainingCupData
from validation import Validator

X, y = readTrainingCupData("data/cup/ML-CUP23-TR.csv")

#take only 50% of test data
X = X[:int(len(X)*0.5)] #WARNING: do not touch this line
y = y[:int(len(y)*0.5)] #WARNING: do not touch this line


#####################################################################
# during experimentig phase apply changese only from here and below #
# | |                     |         |                           | | #
# V V                     V         V                           V V #
#####################################################################



standardInit = False
nn_cup = NeuralNet([LayerDense(10, 30, ActivationTanH(standardInit=standardInit)),
                    LayerDense(30, 25, ActivationTanH(standardInit=standardInit)),
                    LayerDense(25, 20, ActivationTanH(standardInit=standardInit)),
                    LayerDense(20, 15, ActivationTanH(standardInit=standardInit)),
                    LayerDense(15, 3, ActivationLinear())
                    ])

validator = Validator(nn_cup, X, y, MEE, showPlot=False)#important: use MEE as loss function
trErr, valErr, trErrDev, valErrDev, _, _ = validator.kfold(k=20,
                                                            epochs=200,
                                                            lambdaRegularization=0.0,
                                                            patience=-1,
                                                            r_prop=RProp(),
                                                            outputProcessor=DataProcessor(y, standardize=True, independentColumns=True)
                                                            )

print("|--------- Standardize Data ---------|")
print("TrErr: ", trErr)
print("ValErr: ", valErr)
print("TrErrDev: ", trErrDev)
print("ValErrDev: ", valErrDev)

trErr, valErr, trErrDev, valErrDev, _, _ = validator.kfold(k=20,
                                                            epochs=200,
                                                            lambdaRegularization=0.0,
                                                            patience=-1,
                                                            r_prop=RProp(),
                                                            outputProcessor=DataProcessor(y, normalize=True, independentColumns=True)
                                                            )


print("|---------  Normalize Data  ---------|")
print("TrErr: ", trErr)
print("ValErr: ", valErr)
print("TrErrDev: ", trErrDev)
print("ValErrDev: ", valErrDev)

trErr, valErr, trErrDev, valErrDev, _, _ = validator.kfold(k=20,
                                                            epochs=200,
                                                            lambdaRegularization=0.0,
                                                            patience=-1,
                                                            r_prop=RProp(),
                                                            outputProcessor=None
                                                            )


print("|---------     Raw Data     ---------|")
print("TrErr: ", trErr)
print("ValErr: ", valErr)
print("TrErrDev: ", trErrDev)
print("ValErrDev: ", valErrDev)