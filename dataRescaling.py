import numpy as np


class DataProcessor:
    def __init__(self, data:np.ndarray, normalize:bool = None, standardize:bool = None, independentColumns:bool = True):
        
        if normalize is None and standardize is None:
            raise Exception("You must specify at least one of normalize or standardize")
        if normalize is not None and standardize is not None:
            raise Exception("You can't specify both normalize and standardize")
        
        self.normalize = normalize
        self.standardize = standardize
        self.independentColumns = independentColumns
        
        if independentColumns and data.ndim == 2:
            self.processors:list(DataProcessor) = []
            for i in range(data.shape[1]):
                self.processors.append(DataProcessor(data[:,i], normalize, standardize, False))
        elif not independentColumns or data.ndim == 1:
            self.max = np.max(data)
            self.min = np.min(data)
            self.avg = np.mean(data)
            self.std = np.std(data)


    def reset(self, data:np.ndarray):
        self.__init__(data, self.normalize, self.standardize, self.independentColumns)

    def _normalize(self, data:np.ndarray):
        if self.independentColumns:
            for i in range(data.shape[1]):
                data[:,i] = self.processors[i]._normalize(data[:,i])
            return data
        return (data - self.min)/(self.max - self.min)
    
    def _denormalize(self, data:np.ndarray):
        if self.independentColumns:
            for i in range(data.shape[1]):
                data[:,i] = self.processors[i]._denormalize(data[:,i])
            return data
        return data*(self.max - self.min) + self.min
    
    def _standardize(self, data:np.ndarray):
        if self.independentColumns:
            for i in range(data.shape[1]):
                data[:,i] = self.processors[i]._standardize(data[:,i])
            return data
        return (data - self.avg)/self.std
    
    def _destandardize(self, data:np.ndarray):
        if self.independentColumns:
            for i in range(data.shape[1]):
                data[:,i] = self.processors[i]._destandardize(data[:,i])
            return data
        return data*self.std + self.avg

    def process(self, data:np.ndarray) -> np.ndarray:
        if self.normalize: return self._normalize(data)
        if self.standardize: return self._standardize(data)

    def deprocess(self, data:np.ndarray) -> np.ndarray:
        if self.normalize: return self._denormalize(data)
        if self.standardize: return self._destandardize(data)