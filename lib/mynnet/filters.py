class Filter:
    
    def __call__(self, eta, grads):
        raise NotImplementedError()
        
class NoOpFilter:
    
    def __call__(self, deltas):
        return tuple(deltas)
        
class MomentumFilter(Filter):
    
    def __init__(self, gamma):
        self.G = gamma
        self.M = None
        
    def __call__(self, deltas):
        if self.M is None:
            self.M = [p.copy() for p in deltas]
        else:
            self.M = [p*(1.0-self.G) + m*self.G for p, m in zip(deltas, self.M)]
        return tuple(self.M)
        
class AdaDelta(Filter):
    
    def __init__(self, rho = 0.95, epsilon=1.0e-6):
        self.Rho = rho
        self.Epsilon = epsilon
        self.Eg2 = 0.0
        self.Edx2 = 0.0
        
    def __call__(selg, )