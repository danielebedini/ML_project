import numpy as np

def createData(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for classNumber in range(classes):
        ix = range(points*classNumber, points*(classNumber+1))
        r = np.linspace(0.0, 1, points)
        t = np.linspace(classNumber*4, (classNumber+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = classNumber
    return X, y