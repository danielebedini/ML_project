import numpy as np
import matplotlib.pyplot as plt

def generate_data(n_inputs):

    X = []
    y = []

    for i in range(n_inputs):
        x1 = np.random.uniform(-1, 1)
        x2 = np.random.uniform(-1, 1)
        
        X.append([x1, x2])
        # add noise
        #y1 = 1*(x1+x2) + np.random.normal(-0.05, 0.05)
        y2 = np.cos(x1+x2) + np.random.normal(-0.1, 0.1)
        y3 = np.sin(x1-x2) + np.random.normal(-0.1, 0.1)
        y.append([y2, y3])

    X = np.array(X)
    y = np.array(y)
    
#    plt.scatter(X[:, 0], X[:, 1], c=y)
#    plt.show()

    return X, y

