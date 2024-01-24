class LearningRate:
    def __init__(self, learningRate:float):
        self.learningRate = learningRate

    def __call__(self, epoch:int = 0) -> float:
        return self.learningRate

class LearningRateLinearDecay(LearningRate):
    def __init__(self, learningRate:float, tau:int, eta_tau:float):
        self.learningRate = learningRate
        self.tau = tau
        self.eta_tau = eta_tau

    def __call__(self, epoch:int) -> float:
        if self.tau==0:
            return self.learningRate
        if epoch < self.tau:
            alpha = epoch/self.tau
            return (1-alpha)*self.learningRate + alpha*self.eta_tau
        else:
            return self.eta_tau