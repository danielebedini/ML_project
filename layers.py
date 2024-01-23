import numpy as np
from activations import ActivationFunction
from r_prop_parameter import RProp

class LayerDense:

    def __init__(self, nInputs, nNeurons, activationFunction:ActivationFunction):
        self.activationFunction = activationFunction
        self.weights = activationFunction.initialize_weights(nInputs, nNeurons)
        self.bias = 0.05 * np.random.randn(1, nNeurons)
        self.pastGradient = np.zeros_like(self.weights)
        self.rp_delta = None
        self.rp_delta_bias = None
        self.pastGradient_bias = 0


    def reset(self):
        self.weights = self.activationFunction.initialize_weights(self.weights.shape[0], self.weights.shape[1])
        self.bias = 0.05 * np.random.randn(self.bias.shape[0], self.bias.shape[1])
        self.pastGradient = np.zeros_like(self.weights)
        self.rp_delta = None
        self.rp_delta_bias = None
        self.pastGradient_bias = 0


    def forward(self, inputs):
        self.output_previous_layer = inputs
        self.outputNotActivated = np.dot(inputs, self.weights) + self.bias
        self.outputActivated  = self.activationFunction.forward(self.outputNotActivated)
        return self.outputActivated
    

    def backward(self, d_next_layer, learningRate = 0.001, weights_next_layer = None, lambdaRegularization:float = 0, momentum:float = 0.9, r_prop:RProp|None = None):
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
            #reshape gradient if needed e.g from (3,) to (3,1)
            if self.weights.shape[1] == 1: self.gradient = self.gradient.reshape(self.weights.shape)
            if r_prop:
                self.rp_delta, self.pastGradient = r_prop.update_layer(self.gradient, self.pastGradient, self.rp_delta)
                self.weights -= self.pastGradient + lambdaRegularization*self.weights
                self.rp_delta_bias, self.pastGradient_bias = r_prop.update_layer(np.sum(self.delta, axis=0, keepdims=True), self.pastGradient_bias, self.rp_delta_bias)
                self.bias -= self.pastGradient_bias + lambdaRegularization*self.bias
            else:
                self.gradient = np.clip(self.gradient, -0.5, 0.5)
                # gradient        = multiply by learning rate  +          regularization           +       momentum
                self.pastGradient = learningRate*self.gradient + lambdaRegularization*self.weights + self.pastGradient*momentum
                self.weights -= self.pastGradient
                # new bias = gradient descent*learning rate
                self.bias -= np.sum(learningRate*self.delta, axis=0, keepdims=True)

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
            if r_prop:
                self.rp_delta, self.pastGradient = r_prop.update_layer(self.gradient, self.pastGradient, self.rp_delta)
                self.weights -= self.pastGradient + lambdaRegularization*self.weights
                self.rp_delta_bias, self.pastGradient_bias = r_prop.update_layer(np.sum(self.delta, axis=0, keepdims=True), self.pastGradient_bias, self.rp_delta_bias)
                self.bias -= self.pastGradient_bias + lambdaRegularization*self.bias
            else: 
                self.gradient = np.clip(self.gradient, -1, 1)
                # gradient        = multiply by learning rate  +          regularization           +       momentum
                self.pastGradient = learningRate*self.gradient + lambdaRegularization*self.weights + self.pastGradient*momentum
                self.weights -= self.pastGradient
                # new bias = gradient descent*learning rate
                self.bias -= np.sum(learningRate*self.delta, axis=0, keepdims=True)

