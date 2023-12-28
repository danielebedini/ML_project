from loss import LossMSE
from layers import *
from data import createData

# Create data
X, y = createData(100, 3)

layer1 = LayerDense(2, 5, ActivationReLU())
layer2 = LayerDense(5, 2, ActivationReLU())
layer3 = LayerDense(2, 1, ActivationLinear())

for epoch in range(1000):
    layer1.forward(X)
    layer2.forward(layer1.outputActivated)
    layer3.forward(layer2.outputActivated)
    loss = LossMSE(y, layer3.outputActivated)
    print("*************")
    #print(loss)
    #compute ouutput - expected output using numpy
    diff = np.subtract(layer3.outputActivated.T, y).T
    layer3.backward(diff)
    layer2.backward(layer3.dcurrent, layer3.weights)
    layer1.backward(layer2.dcurrent, layer2.weights)