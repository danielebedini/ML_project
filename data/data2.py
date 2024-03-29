import matplotlib.pyplot as plt
import tests.nnfs as nnfs
from nnfs.datasets import vertical_data

nnfs.init()

X, y = vertical_data(samples=100, classes=2)

plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')
plt.show()