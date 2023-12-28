import numpy as np

class LayerDense:
    def __init__(self, nInputs, nNeurons, activationFunction):
        self.weights = 0.10 * np.random.randn(nInputs, nNeurons)
        self.activationFunction = activationFunction

    def forward(self, inputs):
        print(inputs.shape)
        print(self.weights.shape)
        print("--------------")
        self.outputNotActivated = np.dot(inputs, self.weights)
        self.outputActivated  = self.activationFunction.forward(self.outputNotActivated)
        return self.outputActivated
    
    def backward(self, dvalues, learningRate = 0.001, weights_next_layer = None):
        '''
        dvalues: the sigma of the next layer
        '''
        print(str(dvalues.shape) + " dvalues")
        derivative = self.activationFunction.derivative(self.outputNotActivated)
        print(str(self.outputActivated.shape) + " output")
        print(str(derivative.shape) + " derivative")
        
        #this is not wrong but dvalues should be computed differently
        #by computing the sum of dvalues * weights_next_layer
        if weights_next_layer is not None:
            dvalues = np.dot(dvalues, weights_next_layer.T)
            print(str(dvalues.shape) + " dvalues")

        self.dcurrent = np.multiply(dvalues, derivative)
        print(str(self.dcurrent.shape) + " dcurrent")

        self.dweights = np.multiply(self.dcurrent, self.outputActivated)
        print(str(self.dweights.shape) + " dweights")
        print(str(self.weights.sum(axis=0).shape) + " weights sum")
        print(self.weights.sum(axis=0))
        
        self.weights -= learningRate*self.dweights.sum(axis=0)
        print("------------------")
        return self.dcurrent
        


    
class ActivationReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        return self.output

    def derivative(self, inputs):
        '''
        this function calculates  f'(w.T*x)
        '''
        self.dinputs = np.copy(inputs)
        self.dinputs[self.dinputs <= 0] = 0
        self.dinputs[self.dinputs > 0] = 1
        return self.dinputs
    
class ActivationLinear:
    def forward(self, inputs):
        self.output = inputs
        return self.output
    
    def derivative(self, inputs):
        '''
        this function calculates  f'(w.T*x)
        '''
        self.dinputs = np.ones_like(inputs)
        return self.dinputs