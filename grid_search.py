import json
import time
import numpy as np
from dataRescaling import DataProcessor
from learningRate import LearningRate

from metrics import MEE, rSquare
from r_prop_parameter import RProp
from validation import Validator
from utilities import printProgressBar


def grid_search_RProp(X:np.ndarray, y:np.ndarray, folds:int, hyperParameters:{str:list}):
    '''
    data: the data to be used for the grid search
    folds: the number of folds for the kfold validation
    hyperParameters: a dictionary of hyperparameters to be tested

    hyperParameter MUST contain:
        - delta_0
        - delta_max
        - at least one model
        - the maximum number of epochs
    '''
    check_hyperparameters(hyperParameters)
    results = []
    i = 1
    startTime = time.time()
    file = open('grid_search_results_RProp.json', 'a')
    file.write('[\n')
    file.close()
    totIterations = len(hyperParameters['model'])*len(hyperParameters['delta_0'])*len(hyperParameters['delta_max'])*len(hyperParameters['lambdaRegularization'])*len(hyperParameters['preprocess'])*len(hyperParameters['standardInit'])
    printProgressBar(i, totIterations, prefix = 'Progress:', suffix = 'ETA -m -s', length = 50)
    for model in hyperParameters['model']:
        validator = Validator(model, X, y, MEE, rSquare, showPlot=False)
        for d0 in hyperParameters['delta_0'] if 'delta_0' in hyperParameters else [0.1]:
            for dm in hyperParameters['delta_max'] if 'delta_max' in hyperParameters else [50]:
                for reg in hyperParameters['lambdaRegularization'] if 'lambdaRegularization' in hyperParameters else [0]:
                    if reg > 0:
                        patience = 10
                    else:
                        patience = -1
                    for preprocess in hyperParameters['preprocess'] if 'preprocess' in hyperParameters else [None]:
                        if preprocess is not None:
                            if preprocess == 'standardize':
                                preprocessor = DataProcessor(y, standardize=True, independentColumns=True)
                            elif preprocess == 'normalize':
                                preprocessor = DataProcessor(y, normalize=True, independentColumns=True)
                            else:
                                preprocessor = None
                        for standardInit in hyperParameters['standardInit'] if 'standardInit' in hyperParameters else [False]:
                            model.reset(standardInit=standardInit)
                            trE, valE, trErrDev, valErrDev, trA, valA, trAccDev, valAccDev = validator.kfold(k=folds,
                                                                                                        epochs=hyperParameters['epochs'],
                                                                                                        lambdaRegularization=reg,
                                                                                                        patience=patience,
                                                                                                        r_prop=RProp(delta_0=d0, delta_max=dm),
                                                                                                        outputProcessor=preprocessor
                                                                                                        )
                            results = {
                                    'model': model.name,
                                    'standardInit': standardInit,
                                    'delta_0': d0,
                                    'delta_max': dm,
                                    'lambdaRegularization': reg,
                                    'preprocess': preprocess,
                                    'trE': trE,
                                    'valE': valE,
                                    'trErrDev': trErrDev,
                                    'valErrDev': valErrDev,
                                    'trA': trA,
                                    'valA': valA,
                                    'trAccDev': trAccDev,
                                    'valAccDev': valAccDev
                                }
                            i += 1
                            currTime = time.time()
                            elapsedTime = currTime - startTime
                            eta = elapsedTime/i*(totIterations-i)
                            minutes = int(eta/60)
                            seconds = int(eta%60)
                            file = open('grid_search_results_RProp.json', 'a')
                            printProgressBar(i, totIterations, prefix = 'Progress:', suffix = f'ETA {minutes}m {seconds}s  ', length = 50)
                            file.write(json.dumps(results, indent=4))
                            if i != totIterations + 1:
                                file.write(',\n')
                            file.close()
    file = open('grid_search_results_RProp.json', 'a')
    file.write('\n]')
    file.close()

def grid_search_momentum(X:np.ndarray, y:np.ndarray, folds:int, hyperParameters:{str:list}):
    '''
    data: the data to be used for the grid search
    folds: the number of folds for the kfold validation
    hyperParameters: a dictionary of hyperparameters to be tested

    hyperParameter MUST contain:
        - momentum
        - at least one model
        - the maximum number of epochs
    '''
    check_hyperparameters(hyperParameters)
    results = []
    i = 1
    startTime = time.time()
    file = open('grid_search_results_momentum.json', 'a')
    file.write('[\n')
    file.close()
    totIterations = len(hyperParameters['model'])*len(hyperParameters['momentum'])*len(hyperParameters['lambdaRegularization'])*len(hyperParameters['preprocess'])*len(hyperParameters['standardInit']*len(hyperParameters['batch_size'])*len(hyperParameters['learningRate']))
    printProgressBar(i, totIterations, prefix = 'Progress:', suffix = 'ETA -m -s', length = 50)
    for model in hyperParameters['model']:
        validator = Validator(model, X, y, MEE, rSquare, showPlot=False)
        for momentum in hyperParameters['momentum'] if 'momentum' in hyperParameters else [0]:
            for learningRate in hyperParameters['learningRate'] if 'learningRate' in hyperParameters else [LearningRate(0.01)]:
                for reg in hyperParameters['lambdaRegularization'] if 'lambdaRegularization' in hyperParameters else [0]:
                    if reg > 0:
                        patience = 10
                    else:
                        patience = -1
                    for preprocess in hyperParameters['preprocess'] if 'preprocess' in hyperParameters else [None]:
                        if preprocess is not None:
                            if preprocess == 'standardize':
                                preprocessor = DataProcessor(y, standardize=True, independentColumns=True)
                            elif preprocess == 'normalize':
                                preprocessor = DataProcessor(y, normalize=True, independentColumns=True)
                            else:
                                preprocessor = None
                        for standardInit in hyperParameters['standardInit'] if 'standardInit' in hyperParameters else [False]:
                            for batch_size in hyperParameters['batch_size'] if 'batch_size' in hyperParameters else [1]:
                                model.reset(standardInit=standardInit)
                                trE, valE, trErrDev, valErrDev, trA, valA, trAccDev, valAccDev = validator.kfold(k=folds,
                                                                                                            epochs=hyperParameters['epochs'],
                                                                                                            lambdaRegularization=reg,
                                                                                                            patience=patience,
                                                                                                            learningRate=learningRate,
                                                                                                            momentum=momentum,
                                                                                                            batch_size=batch_size,
                                                                                                            outputProcessor=preprocessor
                                                                                                            )
                                results = {
                                        'model': model.name,
                                        'standardInit': standardInit,
                                        'momentum': momentum,
                                        'learningRate': learningRate.learningRate,
                                        'lambdaRegularization': reg,
                                        'preprocess': preprocess,
                                        'batch_size': batch_size,
                                        'trE': trE,
                                        'valE': valE,
                                        'trErrDev': trErrDev,
                                        'valErrDev': valErrDev,
                                        'trA': trA,
                                        'valA': valA,
                                        'trAccDev': trAccDev,
                                        'valAccDev': valAccDev
                                    }
                                i += 1
                                currTime = time.time()
                                elapsedTime = currTime - startTime
                                eta = elapsedTime/i*(totIterations-i)
                                minutes = int(eta/60)
                                seconds = int(eta%60)
                                file = open('grid_search_results_momentum.json', 'a')
                                printProgressBar(i, totIterations, prefix = 'Progress:', suffix = f'ETA {minutes}m {seconds}s  ', length = 50)
                                file.write(json.dumps(results, indent=4))
                                if i != totIterations + 1:
                                    file.write(',\n')
                                file.close()
    file = open('grid_search_results_momentum.json', 'a')
    file.write('\n]')
    file.close()
    
def random_search_RProp(X:np.ndarray, y:np.ndarray, folds:int, iterations:int , hyperParameters:{str:list}):
    '''
    data: the data to be used for the grid search
    folds: the number of folds for the kfold validation
    hyperParameters: a dictionary of hyperparameters to be tested

    hyperParameter MUST contain:
        - delta_0
        - delta_max
        - at least one model
        - the maximum number of epochs
    '''
    check_hyperparameters(hyperParameters)
    results = []
    i = 1
    startTime = time.time()
    file = open('random_search_results_RProp.json', 'a')
    file.write('[\n')
    file.close()
    for _ in range(iterations):
        model = np.random.choice(hyperParameters['model'])
        preprocess = np.random.choice(hyperParameters['preprocess'])
        if preprocess == 'standardize':
            preprocessor = DataProcessor(y, standardize=True, independentColumns=True)
        elif preprocess == 'normalize':
            preprocessor = DataProcessor(y, normalize=True, independentColumns=True)
        else:
            preprocessor = None
        model.reset(standardInit=np.random.choice(hyperParameters['standardInit']))
        reg = hyperParameters['lambdaRegularization']()
        patience = hyperParameters['patience']()
        delta_0 = hyperParameters['delta_0']()
        delta_max = hyperParameters['delta_max']()
        validator = Validator(model, X, y, MEE, rSquare, showPlot=False)
        trE, valE, trErrDev, valErrDev, trA, valA, trAccDev, valAccDev = validator.kfold(k=folds,
                                                                                        epochs=hyperParameters['epochs'],
                                                                                        lambdaRegularization=reg,
                                                                                        patience=patience,
                                                                                        r_prop=RProp(delta_0=delta_0, delta_max=delta_max),
                                                                                        outputProcessor=preprocessor
                                                                                        )
        results = {
                'model': model.name,
                'delta_0': delta_0,
                'delta_max': delta_max,
                'lambdaRegularization': reg,
                'preprocess': preprocess,
                'trE': trE,
                'valE': valE,
                'trErrDev': trErrDev,
                'valErrDev': valErrDev,
                'trA': trA,
                'valA': valA,
                'trAccDev': trAccDev,
                'valAccDev': valAccDev
            }
        i += 1
        currTime = time.time()
        elapsedTime = currTime - startTime
        eta = elapsedTime/i*(iterations-i)
        minutes = int(eta/60)
        seconds = int(eta%60)
        file = open('random_search_results_RProp.json', 'a')
        printProgressBar(i, iterations, prefix = 'Progress:', suffix = f'ETA {minutes}m {seconds}s  ', length = 50)
        file.write(json.dumps(results, indent=4))
        if i != iterations + 1:
            file.write(',\n')
        file.close()
    file = open('random_search_results_RProp.json', 'a')
    file.write('\n]')
    file.close()


def check_hyperparameters(hyperParameters:{str:list}):
    if 'model' not in hyperParameters:
        raise Exception('You must specify at least one model')
    if 'epochs' not in hyperParameters:
        raise Exception('You must specify the maximum number of epochs')