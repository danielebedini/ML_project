from activations import *
from dataRescaling import DataProcessor
from layers import LayerDense
from learningRate import LearningRate
from metrics import MEE, LossMSE, rSquare
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

'''
dataProcessor = DataProcessor(y, standardize=True, independentColumns=True)
y = dataProcessor.process(y)
nn_cup.train(X, y,
            epochs=200,
            r_prop=RProp(),
            patience=-1,
            lambdaRegularization=0.0,
            )

#evaluate
y_pred = nn_cup.forward(X)
y_pred = dataProcessor.deprocess(y_pred)
y = dataProcessor.deprocess(y)
print("MEE: ", MEE(y, y_pred))
print("R Squared: ", rSquare(y, y_pred))
'''

validator = Validator(nn_cup, X, y, MEE, rSquare, showPlot=False)#important: use MEE as loss function
trErr, valErr, trErrDev, valErrDev, trAcc, valAcc, trAccDev, valAccDev = validator.kfold(k=20,
                                                                                        epochs=200,
                                                                                        lambdaRegularization=0.0,
                                                                                        patience=-1,
                                                                                        r_prop=RProp(),
                                                                                        outputProcessor=DataProcessor(y, standardize=True, independentColumns=True)
                                                                                        )

print("|--------- Standardize Data ---------|")
print(f"TrErr:  {'%.4f' % trErr} ± {'%.4f' % trErrDev}")
print(f"ValErr: {'%.4f' % valErr} ± {'%.4f' % valErrDev}")
print(f"TrAcc:  {'%.4f' % trAcc} ± {'%.4f' % trAccDev}")
print(f"ValAcc: {'%.4f' % valAcc} ± {'%.4f' % valAccDev}")

trErr, valErr, trErrDev, valErrDev, trAcc, valAcc, trAccDev, valAccDev = validator.kfold(k=20,
                                                                                        epochs=200,
                                                                                        lambdaRegularization=0.0,
                                                                                        patience=-1,
                                                                                        r_prop=RProp(),
                                                                                        outputProcessor=DataProcessor(y, normalize=True, independentColumns=True)
                                                                                        )


print("|---------  Normalize Data  ---------|")
print(f"TrErr:  {'%.4f' % trErr} ± {'%.4f' % trErrDev}")
print(f"ValErr: {'%.4f' % valErr} ± {'%.4f' % valErrDev}")
print(f"TrAcc:  {'%.4f' % trAcc} ± {'%.4f' % trAccDev}")
print(f"ValAcc: {'%.4f' % valAcc} ± {'%.4f' % valAccDev}")

trErr, valErr, trErrDev, valErrDev, trAcc, valAcc, trAccDev, valAccDev = validator.kfold(k=20,
                                                                                        epochs=200,
                                                                                        lambdaRegularization=0.0,
                                                                                        patience=-1,
                                                                                        r_prop=RProp(),
                                                                                        outputProcessor=None
                                                                                        )


print("|---------     Raw Data     ---------|")
print(f"TrErr:  {'%.4f' % trErr} ± {'%.4f' % trErrDev}")
print(f"ValErr: {'%.4f' % valErr} ± {'%.4f' % valErrDev}")
print(f"TrAcc:  {'%.4f' % trAcc} ± {'%.4f' % trAccDev}")
print(f"ValAcc: {'%.4f' % valAcc} ± {'%.4f' % valAccDev}")