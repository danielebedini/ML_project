import numpy as np

class LayerDense:

    def __init__(self, nInputs, nNeurons, activationFunction):
        self.weights = 0.10 * np.random.randn(nInputs, nNeurons)
        self.activationFunction = activationFunction

    def forward(self, inputs):
        self.output_previous_layer = inputs
        self.outputNotActivated = np.dot(inputs, self.weights)
        self.outputActivated  = self.activationFunction.forward(self.outputNotActivated)
        return self.outputActivated
    
    def backward(self, d_next_layer, learningRate = 0.001, weights_next_layer = None):
        '''
        Output layer:
            - compute d_error (given by input d_next_layer)
            - compute d_activation in respect to the layer inputs
            - compute delta = d_next_layer * d_activation
            - gradient = np.dot(output_previous_layer.T, delta)
            - bias = 1 * delta
            - gradient descent: calculate the new weights
        '''
        if weights_next_layer is None: # Output layer
            #compute d_activation in respect to the layer inputs
            d_activation = self.activationFunction.derivative(self.outputNotActivated)
            #compute delta = d_error * d_activation
            self.delta = d_next_layer * d_activation
            # calculate gradient
            self.gradient = np.dot(self.output_previous_layer.T, self.delta)
            # calculate new weights
            self.weights += learningRate*self.gradient - 0.001*self.weights
            # calculate bias
            #self.bias = 1*self.delta

        '''
        Hidden layer:
            - compute d_activation in respect to the layer inputs
            - compute d_error = np.dot(delta_next_layer, weights_next_layer.T)
            - compute delta = d_error * d_activation
            - gradient = np.dot(output_previous_layer.T, delta)
            - bias = 1 * delta
            - gradient descent: calculate the new weights
        '''
        if weights_next_layer is not None: # Hidden layer
            d_activation = self.activationFunction.derivative(self.outputNotActivated)
            self.d_error = np.dot(d_next_layer, weights_next_layer.T)
            self.delta = self.d_error * d_activation
            self.gradient = np.dot(self.output_previous_layer.T, self.delta)
            #self.weights += -learningRate*self.gradient - 0.001*self.weights
            #self.bias = 1*self.delta

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
    
class ActivationTanH:
    def forward(self, inputs):
        self.output = np.tanh(inputs)
        return self.output
    
    def derivative(self, inputs):
        self.dinputs = 1 - self.output**2
        return self.dinputs