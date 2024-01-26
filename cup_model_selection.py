from activations import *
from dataRescaling import DataProcessor
from grid_search import grid_search_RProp, grid_search_momentum, random_search_RProp
from layers import LayerDense
from learningRate import LearningRate
from metrics import MEE, LossMSE, rSquare
from net import NeuralNet
from r_prop_parameter import RProp
from utilities import plot_data_error, readTrainingCupData
from validation import Validator

X, y = readTrainingCupData("data/cup/ML-CUP23-TR.csv")

#take only 75% of the data for training and validation (the remaining 25% will be used for testing)
X = X[:int(len(X)*0.75)] #WARNING: do not touch this line
y = y[:int(len(y)*0.75)] #WARNING: do not touch this line


#####################################################################
# during experimentig phase apply changese only from here and below #
# | |                     |         |                           | | #
# V V                     V         V                           V V #
#####################################################################




nn_cup1 = NeuralNet([LayerDense(10, 30, ActivationTanH()),
                    LayerDense(30, 15, ActivationTanH()),
                    LayerDense(15, 3, ActivationLinear())
                    ],
                    name="tanH_30_15_3")

nn_cup2 = NeuralNet([LayerDense(10, 40, ActivationSigmoid()),
                    LayerDense(40, 3, ActivationLinear())
                    ],
                    name="sigmoid_40_3")

nn_cup3 = NeuralNet([LayerDense(10, 40, ActivationTanH()),
                    LayerDense(40, 3, ActivationLinear())
                    ],
                    name="tanH_40_3")


'''
grid_search_RProp(X, y, 10,
                  {
                        'epochs': 500,
                        'model': [nn_cup1, nn_cup2, nn_cup3],
                        'delta_0': [0.01, 0.05],
                        'delta_max': [10],
                        'lambdaRegularization': [0.0008, 0.0009, 0.001, 0.002],
                        'preprocess': ['standardize', 'normalize', None],
                        'standardInit': [True, False]
                  }
)
'''
'''
grid_search_momentum(X, y, 10,
                    {
                            'epochs': 4000,
                            'model': [nn_cup1, nn_cup2, nn_cup3],
                            'momentum': [0.9, 0.95, 0.99],
                            'learningRate': [LearningRate(0.3), LearningRate(0.2), LearningRate(0.1)],
                            'batch_size': [80, 110, 120],
                            'lambdaRegularization': [0.0007, 0.0008, 0.0009, 0.001],
                            'preprocess': ['standardize'],
                            'standardInit': [True, False]
                    }
    )
'''
random_search_RProp(X, y, 10, 400,
                    {
                        'epochs': 500,
                        'model': [nn_cup1, nn_cup2, nn_cup3],
                        'delta_0': lambda: np.random.uniform(0.01, 0.1),
                        'delta_max': lambda: np.random.uniform(10, 20),
                        'lambdaRegularization': lambda: np.random.uniform(0.0008, 0.001),
                        'patience': lambda: np.random.randint(5, 13),
                        'preprocess': ['standardize', 'normalize', None],
                        'standardInit': [True, False]
                    }
    )
exit()
validator = Validator(nn_cup3, X, y, MEE, rSquare, showPlot=True)#important: use MEE as loss function
trErr, valErr, trErrDev, valErrDev, trAcc, valAcc, trAccDev, valAccDev = validator.kfold(k=2,
                                                                                        epochs=4000,
                                                                                        lambdaRegularization=0.0008,
                                                                                        patience=10,
                                                                                        momentum=0.9,
                                                                                        learningRate=LearningRate(0.1),
                                                                                        outputProcessor=DataProcessor(y, standardize=True, independentColumns=True)
                                                                                        )

print("|--------- Standardize Data ---------|")
print(f"TrErr:  {'%.4f' % trErr} ± {'%.4f' % trErrDev}")
print(f"ValErr: {'%.4f' % valErr} ± {'%.4f' % valErrDev}")
print(f"TrAcc:  {'%.4f' % trAcc} ± {'%.4f' % trAccDev}")
print(f"ValAcc: {'%.4f' % valAcc} ± {'%.4f' % valAccDev}")
exit()
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


