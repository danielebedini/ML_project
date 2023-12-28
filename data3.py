import numpy as np
import matplotlib.pyplot as plt

def generate_data(n_inputs):

    X = []
    y = []

    for i in range(n_inputs):
        x1 = np.random.randint(-10, 10)
        x2 = np.random.randint(-10, 10)
        
        X.append([x1, x2])
        y.append(x1+x2)

    X = np.array(X)
    y = np.array(y)
    
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()

    return X, y

