import numpy as np


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def readMonkData(filename:str) -> (np.ndarray, np.ndarray):
    input = []
    output = []
    with open(filename, "r") as file:
        for line in file:
            values = list(map(int, line.strip().split()[0:-1]))
            input.append(values[1:])
            output.append(values[0])
    return np.array(input), np.array(output)

def feature_one_hot_encoding(y:np.ndarray, classes:[int]) -> np.ndarray:
    '''
    y: array of labels
    classes: array of number of classes
    '''
    one_hot_len = np.sum(classes)
    one_hot = np.zeros((y.shape[0], one_hot_len))
    for i in range(y.shape[0]): #for each sample
        for j in range(len(classes)): #for each label
            #if y has a single dimension, then we have a single output
            if len(classes) == 1:
                one_hot[i, y[i] - 1] = 1
            else:
                prev_classes = int(np.sum(classes[0:j]))
                one_hot[i, prev_classes + y[i,j] - 1] = 1
    return one_hot

def standard_one_hot_encoding(y:np.ndarray, classes:int) -> np.ndarray:
    '''
    y: array of labels
    classes: number of classes
    '''
    one_hot = np.zeros((y.shape[0], classes))
    for i in range(y.shape[0]): #for each sample
        one_hot[i, y[i] - 1] = 1
    return one_hot

def plot_data_error(trError:np.ndarray, valError:np.ndarray):
    import matplotlib.pyplot as plt
    plt.plot(trError, label="Training error")
    plt.plot(valError, label="Validation error")
    plt.legend()
    plt.show()