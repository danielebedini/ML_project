import sys
import os

sys.path.append(os.path.join(sys.path[0], '..'))

import time
import numpy as np
from metrics import MEE
from cupModels.cup_model_generate import Ensemble
from utilities import readTrainingCupData

X, y = readTrainingCupData("data/cup/ML-CUP23-TR.csv")

#take only 75% of the data for training and validation (the remaining 25% will be used for testing)
X = X[:int(len(X)*0.75)] #WARNING: do not touch this line
y = y[:int(len(y)*0.75)] #WARNING: do not touch this line

testX = X[int(len(X)*0.75):]
testY = y[int(len(y)*0.75):]

start = time.time()

ensembles = []
for i in range(10):
    ensembles.append(Ensemble(X, y, 10))

end = time.time()
print("Training time: ", end-start)

testErrors = []
for ensemble in ensembles:
    testErrors.append(MEE(testY, ensemble.predict(testX)))

print("Test errors: ", testErrors)
print("Average test error: ", np.mean(testErrors))
print("Test error standard deviation: ", np.std(testErrors))
