import numpy as np


class RProp:
    def __init__(self, delta_0:float = 0.1, 
                 delta_max:float = 1.0, 
                 delta_min:float = 1e-6, 
                 eta_minus:float = 0.5, 
                 eta_plus:float = 1.2):
        self.delta_0 = delta_0
        self.delta_max = delta_max
        self.delta_min = delta_min
        self.eta_minus = eta_minus
        self.eta_plus = eta_plus
    
    def update_layer(self, new_err_der:np.ndarray, old_err_der:np.ndarray, old_delta:np.ndarray) -> (np.ndarray, np.ndarray):
        '''
        everything should be shaped as: (n_inputs, n_neurons)
        new_err_der: the derivative of the new error
        old_err_der: the derivative of the old error
        old_delta:  the delta returned in a previous call to this function
        '''
        if old_err_der is None: # first iteration
            old_err_der = np.zeros(new_err_der.shape)
        if old_delta is None: # first iteration
            old_delta = np.ones(new_err_der.shape)*self.delta_0
        
        # compute product of gradient and past gradient to obtain sign
        signArray = np.multiply(new_err_der, old_err_der)
        signArray[signArray == 0] = 0# sign of gradient or  past gradient is zero
        signArray[signArray > 0] = 1 # sign of gradient and past gradient is same
        signArray[signArray < 0] = -1# sign of gradient and past gradient is different

        new_delta = np.vectorize(self._update_delta)(signArray, old_delta)
        new_gradient = np.vectorize(self._update_gradient)(signArray, old_err_der, new_err_der, new_delta)
        new_gradient[signArray < 0] = 0

        return new_delta, new_gradient 
    
    def _update_delta(self, sign, old_delta):
        if sign > 0:
            new_delta = np.minimum(old_delta*self.eta_plus, self.delta_max)
        elif sign < 0:
            new_delta = np.maximum(old_delta*self.eta_minus, self.delta_min)
        else:
            new_delta = old_delta
        return new_delta
    
    def _update_gradient(self, sign, old_err_der, new_err_der, new_delta):
        if sign > 0:
            weight_update_value = np.sign(new_err_der)*new_delta
        elif sign < 0:
            weight_update_value = old_err_der
        else:
            weight_update_value = np.sign(new_err_der)*new_delta
        return weight_update_value