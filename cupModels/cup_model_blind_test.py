import sys
import os

sys.path.append(os.path.join(sys.path[0], '..'))

import time
from cupModels.cup_model_generate import Ensemble
from utilities import readTestCupData, readTrainingCupData

X, y = readTrainingCupData("data/cup/ML-CUP23-TR.csv")

start = time.time()

#here we train on all the data for the final model
ensemble = Ensemble(X, y, 10)

end = time.time()
print("Training time: ", end-start)

#use the model to predict the blid test
blindX = readTestCupData("data/cup/ML-CUP23-TS.csv")
results = ensemble.predict(blindX)

with open("Diehard_Tikhonov_fanboys_ML-CUP23-TS.csv", 'a') as file:
    file.write("# Davide Borghini, Daniele Bedini\n")
    file.write("# Diehard Tikhonov Fanboys\n")
    file.write("# ML-CUP23\n")
    file.write("# Submission Date 30/01/2024\n")
    for i in range(results.shape[0]):
        file.write(f'{i+1},{results[i][0]},{results[i][1]},{results[i][2]}\n')
