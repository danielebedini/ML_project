import numpy as np

class ActivationFunction:
    def forward(self, inputs):
        raise NotImplementedError
    def derivative(self, inputs):
        raise NotImplementedError

class ActivationReLU(ActivationFunction):
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
    
class ActivationLinear(ActivationFunction):
    def forward(self, inputs):
        self.output = inputs
        return self.output
    
    def derivative(self, inputs):
        '''
        this function calculates  f'(w.T*x)
        '''
        self.dinputs = np.ones_like(inputs)
        return self.dinputs
    
class ActivationTanH(ActivationFunction):
    def forward(self, inputs):
        self.output = np.tanh(inputs)
        return self.output
    
    def derivative(self, inputs):
        self.dinputs = 1 - self.output**2
        return self.dinputs
    
class ActivationSigmoid(ActivationFunction): #TODO: check this
    def forward(self, inputs):
        self.output = 1/(1+np.exp(-inputs))
        return self.output
    
    def derivative(self, inputs):
        self.dinputs = self.output * (1-self.output)
        return self.dinputs
    
class ActivationSoftmax(ActivationFunction):
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self.output
    
    def derivative(self, inputs):
        self.dinputs = np.ones_like(inputs)
        return self.dinputs

class ActivationLeakyReLU(ActivationFunction):
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    
    def forward(self, inputs):
        self.output = np.maximum(self.alpha*inputs, inputs)
        return self.output
    
    def derivative(self, inputs):
        self.dinputs = np.ones_like(inputs)
        self.dinputs[inputs <= 0] = self.alpha
        return self.dinputs
