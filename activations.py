import numpy as np

class ActivationFunction:
    def forward(self, inputs):
        raise NotImplementedError
    
    def derivative(self, inputs):
        raise NotImplementedError
    
    def initialize_weights(self, nInputs, nNeurons):
        print("WARNING: initialized with random weights.")
        return 0.05 * np.random.randn(nInputs, nNeurons)

    def _initialize_xavier(self, nInputs, nNeurons): # Xavier initialization for sigmoid, tanh, ecc.. activation function
        return np.random.randn(nInputs, nNeurons) * np.sqrt(2/(nInputs+nNeurons))
    
    def _initialize_he(self, nInputs, nNeurons): # He initialization for ELU or similar activation function
        return np.random.randn(nInputs, nNeurons) * np.sqrt(2/nInputs)
    

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

    def initialize_weights(self, nInputs, nNeurons):
        return self._initialize_he(nInputs, nNeurons)
    

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
    
    def initialize_weights(self, nInputs, nNeurons):
        return self._initialize_he(nInputs, nNeurons)
    
    
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
    
    def initialize_weights(self, nInputs, nNeurons): 
        return self._initialize_xavier(nInputs, nNeurons)
    
class ActivationSigmoid(ActivationFunction): #TODO: check this
    def forward(self, inputs):
        self.output = 1/(1+np.exp(-inputs))
        return self.output
    
    def derivative(self, inputs):
        self.dinputs = self.output * (1-self.output)
        return self.dinputs
    
    def initialize_weights(self, nInputs, nNeurons):
        return self._initialize_xavier(nInputs, nNeurons)
    
class ActivationSoftmax(ActivationFunction):
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self.output
    
    def derivative(self, inputs):
        self.dinputs = np.ones_like(inputs)
        return self.dinputs
    
    def initialize_weights(self, nInputs, nNeurons):
        return self._initialize_xavier(nInputs, nNeurons)