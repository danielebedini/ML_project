
from layers import *
from net import NeuralNet
from data3 import generate_data
# Create data
X, y = generate_data(400)

'''
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
    layer2.backward(layer3.delta, layer3.weights)
    layer1.backward(layer2.delta, layer2.weights)
'''

nn = NeuralNet([LayerDense(2, 6, ActivationTanH()),
                LayerDense(6, 2, ActivationLinear())])

nn.train(X, y, 0.001, 300, batch_size=-1)

print(nn.forward([[1, 2]]))