import sys
import os
sys.path.append(os.path.join(sys.path[0], '..'))

from threading import Thread
import time
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

hyperParameterDict = {
                        'epochs': 4000,
                        'delta_0': lambda: np.random.uniform(0.06, 0.12),
                        'delta_max': lambda: 50,
                        'lambdaRegularization': lambda: np.random.uniform(0.000005, 0.00002),
                        'patience': lambda: np.random.randint(6, 12),
                        'preprocess': ['standardize', 'normalize', None],
                        'standardInit': [True, False]
                    }


hyperParameterDict2 = hyperParameterDict.copy()
hyperParameterDict2['model'] = [nn_cup1]
thread1 = Thread(target=random_search_RProp, args=(X, y, 10, 100,hyperParameterDict2, 'hyperParSearch/rand_nn_cup1.json'))

hyperParameterDict3 = hyperParameterDict.copy()
hyperParameterDict3['model'] = [nn_cup2]
thread2 = Thread(target=random_search_RProp, args=(X, y, 10, 100,hyperParameterDict3, 'hyperParSearch/rand_nn_cup2.json'))

hyperParameterDict4 = hyperParameterDict.copy()
hyperParameterDict4['model'] = [nn_cup3]
thread3 = Thread(target=random_search_RProp, args=(X, y, 10, 100,hyperParameterDict4, 'hyperParSearch/rand_nn_cup3.json'))

startingTime = time.time()
print("starting time: ", time.strftime("%H:%M:%S", time.localtime(startingTime)))

thread1.start()
thread2.start()
thread3.start()

thread1.join()
thread2.join()
thread3.join()

endingTime = time.time()
print("ending time: ", time.strftime("%H:%M:%S", time.localtime(endingTime)))
print(f'total time: {endingTime-startingTime} seconds')
