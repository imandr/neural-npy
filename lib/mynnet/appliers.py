import numpy as np
import math

class Applier:
    
    def __call__(self, eta, grads):
        raise NotImplementedError()
        
class SimpleApplier:
    
    def __call__(self, eta, *grads):
        return tuple([-eta*g for g in grads])
        
class MomentumApplier(Applier):
    
    def __init__(self, gamma = 0.95):
        self.G = gamma
        self.M = None
        
    def __call__(self, eta, grads):
        if self.M is None:
            self.M = [g.copy() for g in grads]
        else:
            self.M = [g*(1.0-self.G) + m*self.G for g, m in zip(grads, self.M)]
        return tuple([-eta*g for g in self.M])
        
class NAGApplier(Applier):
    
    def __init__(self, gamma = 0.5):
        self.G = gamma
        self.Dt1 = None      # D(t-1)

    def __call__(self, eta, grads):
        if self.Dt1 is None:
            self.Dt1 = [np.zeros_like(g) for g in grads]
        
        Dt = [self.G * dt1 - eta * g for dt1, g in zip(self.Dt1, grads)]
        deltas = [dt + self.G*(dt-dt1) for dt, dt1 in zip(Dt, self.Dt1)]
        self.Dt1 = Dt
        return tuple(deltas)
        
        
class AdaDeltaApplier(Applier):
    
    def __init__(self, rho = 0.95, epsilon=1.0e-6):
        self.Rho = rho
        self.Epsilon = epsilon
        self.Eg2s = None
        self.Edw2s = None
        self.EtaMA = None
        
    def __call__(self, eta, grads):
        if self.Eg2s is None:
            # first time
            self.Eg2s = [np.sum(np.square(g)) for g in grads]
            dws = [-eta*g for g in grads]
            self.Edw2s = [np.sum(np.square(d)) for d in dws]
            self.EtaMA = np.array([eta for g in grads])
        else:
            g2s = [np.sum(np.square(g)) for g in grads]
            self.Eg2s = [oldG*self.Rho + newG*(1.0-self.Rho) for oldG, newG in zip(self.Eg2s, g2s)]
            es = [math.sqrt((Edw2 + self.Epsilon*eta)/(Eg2 + self.Epsilon)) for Edw2, Eg2 in zip(self.Edw2s, self.Eg2s)]
            dws = [-e*g for e, g in zip(es, grads)]
            self.Edw2s = [oldDw*self.Rho + np.sum(np.square(newDw))*(1.0-self.Rho) for oldDw, newDw in zip(self.Edw2s, dws)]
            self.EtaMA = 0.99*self.EtaMA + 0.01*np.array(es)
            #print "AdaDelta: es=", es, "    eta ma=", self.EtaMA
        return tuple(dws)
