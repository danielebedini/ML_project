import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))

from threading import Thread
import time
from activations import *
from dataRescaling import DataProcessor
from grid_search import grid_search_RProp
from layers import LayerDense
from learningRate import LearningRate
from metrics import MEE, LossMSE, rSquare
from net import NeuralNet
from r_prop_parameter import RProp
from utilities import readTrainingCupData
from validation import Validator

X, y = readTrainingCupData("data/cup/ML-CUP23-TR.csv")

#take only 75% of the data for training and validation (the remaining 25% will be used for testing)
X = X[:int(len(X)*0.75)] #WARNING: do not touch this line
y = y[:int(len(y)*0.75)] #WARNING: do not touch this line

nn_cup1_1 = NeuralNet([LayerDense(10, 30, ActivationTanH()),
                    LayerDense(30, 15, ActivationTanH()),
                    LayerDense(15, 3, ActivationLinear())
                    ],
                    name="tanH_30_15_3")

nn_cup1_2 = NeuralNet([LayerDense(10, 30, ActivationTanH()),
                    LayerDense(30, 15, ActivationTanH()),
                    LayerDense(15, 3, ActivationLinear())
                    ],
                    name="tanH_30_15_3")

nn_cup1_3 = NeuralNet([LayerDense(10, 30, ActivationTanH()),
                    LayerDense(30, 15, ActivationTanH()),
                    LayerDense(15, 3, ActivationLinear())
                    ],
                    name="tanH_30_15_3")

nn_cup1_4 = NeuralNet([LayerDense(10, 30, ActivationTanH()),
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

hyperParameterDict = {
                    'epochs': 4000,
                    'lambdaRegularization': [0.000009, 0.00001, 0.000013, 0.000015],
                    'delta_0': [0.07, 0.075, 0.08, 0.09],
                    'delta_max': [50],
                    'preprocess': ['standardize'],
                    'standardInit': [True]
                    }

#'patience': [10, 11, 12, 13],

hyperParameterDict2 = hyperParameterDict.copy()
hyperParameterDict2['model'] = [nn_cup1_1]
hyperParameterDict2['patience'] = [10]
thread1 = Thread(target=grid_search_RProp, args=(X, y, 10, hyperParameterDict2, 'hyperParSearch/grid_2_nn_cup1.json'))

hyperParameterDict3 = hyperParameterDict.copy()
hyperParameterDict3['model'] = [nn_cup1_2]
hyperParameterDict3['patience'] = [11]
thread2 = Thread(target=grid_search_RProp, args=(X, y, 10, hyperParameterDict3, 'hyperParSearch/grid_2_nn_cup2.json'))

hyperParameterDict4 = hyperParameterDict.copy()
hyperParameterDict4['model'] = [nn_cup1_3]
hyperParameterDict4['patience'] = [12]
thread3 = Thread(target=grid_search_RProp, args=(X, y, 10, hyperParameterDict4, 'hyperParSearch/grid_2_nn_cup3.json'))

hyperParameterDict5 = hyperParameterDict.copy()
hyperParameterDict5['model'] = [nn_cup1_4]
hyperParameterDict5['patience'] = [13]
thread4 = Thread(target=grid_search_RProp, args=(X, y, 10, hyperParameterDict5, 'hyperParSearch/grid_2_nn_cup4.json'))

#pritn("current time")
startingTime = time.time()
print("starting time: ", time.strftime("%H:%M:%S", time.localtime(startingTime)))

thread1.start()
thread2.start()
thread3.start()
thread4.start()

thread1.join()
thread2.join()
thread3.join()
thread4.join()

endingTime = time.time()
print("ending time: ", time.strftime("%H:%M:%S", time.localtime(endingTime)))
print(f'total time: {endingTime-startingTime} seconds')

exit()