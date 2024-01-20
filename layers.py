import numpy as np

class LayerDense:

    def __init__(self, nInputs, nNeurons, activationFunction):
        self.weights = 0.05 * np.random.randn(nInputs, nNeurons)
        self.activationFunction = activationFunction
        self.pastGradient = 0
        self.bias = 0.05 * np.random.randn(1, nNeurons)


    def reset(self):
        self.weights = 0.05 * np.random.randn(self.weights.shape[0], self.weights.shape[1])
        self.bias = 0.05 * np.random.randn(self.bias.shape[0], self.bias.shape[1])
        self.pastGradient = 0


    def forward(self, inputs):
        self.output_previous_layer = inputs
        self.outputNotActivated = np.dot(inputs, self.weights) + self.bias
        self.outputActivated  = self.activationFunction.forward(self.outputNotActivated)
        return self.outputActivated
    
    def backward(self, d_next_layer, learningRate = 0.001, weights_next_layer = None, lambdaRegularization:float = 0, momentum:float = 0.9):
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
            d_activation = self.activationFunction.derivative(self.outputNotActivated)
            #if second shape is 1, then we have a single output
            if d_activation.shape[1] == 1: d_activation = np.reshape(d_activation, d_activation.shape[0])
            self.delta = d_next_layer * d_activation
            # calculate gradient
            self.gradient = np.dot(self.output_previous_layer.T, self.delta)
            self.gradient = np.clip(self.gradient, -0.5, 0.5)
            #reshape gradient if needed e.g from (3,) to (3,1)
            if self.weights.shape[1] == 1: self.gradient = self.gradient.reshape(self.weights.shape)
            # newWeights -= multiply by learning rate  +          regularization           +       momentum
            self.weights -= learningRate*self.gradient + lambdaRegularization*self.weights + self.pastGradient*momentum
            self.pastGradient = self.gradient*learningRate                      # if added regualrized with own term
            # new bias = gradient descent*learning rate                         #+      regularization for bias
            self.bias -= np.sum(learningRate*self.delta, axis=0, keepdims=True) #+ lambdaRegularization*self.bias

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
            #if next layer has a single output, then we have to reshape d_next_layer
            if len(d_next_layer.shape) == 1: d_next_layer = d_next_layer.reshape(d_next_layer.shape[0], 1)
            self.d_error = np.dot(d_next_layer, weights_next_layer.T)
            self.delta = self.d_error * d_activation
            self.gradient = np.dot(self.output_previous_layer.T, self.delta)
            #use only one of the following 2 lines (second is more appropriate for deep networks)
            self.gradient = np.clip(self.gradient, -1, 1)
            #self.gradient = np.sign(self.gradient)
            # newWeights -= multiply by learning rate  +          regularization           +       momentum
            self.weights -= learningRate*self.gradient + lambdaRegularization*self.weights + self.pastGradient*momentum
            self.pastGradient = self.gradient*learningRate                      # if added regualrized with own term
            # new bias = gradient descent*learning rate                         #+      regularization for bias
            self.bias -= np.sum(learningRate*self.delta, axis=0, keepdims=True) #+ lambdaRegularization*self.bias


# ------------- Activation functions -------------


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
    
class ActivationSigmoid: #TODO: check this
    def forward(self, inputs):
        self.output = 1/(1+np.exp(-inputs))
        return self.output
    
    def derivative(self, inputs):
        self.dinputs = self.output * (1-self.output)
        return self.dinputs
    
class ActivationSoftmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self.output
    
    def derivative(self, inputs):
        self.dinputs = np.ones_like(inputs)
        return self.dinputs

class ActivationLeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    
    def forward(self, inputs):
        self.output = np.maximum(self.alpha*inputs, inputs)
        return self.output
    
    def derivative(self, inputs):
        self.dinputs = np.ones_like(inputs)
        self.dinputs[inputs <= 0] = self.alpha
        return self.dinputs
