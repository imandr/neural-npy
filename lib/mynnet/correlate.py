import numpy as np
from mynnet import Layer
import math

class Correlate(Layer):
  
    def __init__(self, u, v, out_size, weight_decay=1.e-5,
                name = None, applier = None):
        Layer.__init__(self, [u,v], name)
        self.U = u
        self.V = v
        self.NU = self.U.shape[0]
        self.NV = self.V.shape[0]
        self.Nout = out_size
        self.WeightDecay = weight_decay
        self.OutShape = (out_size,)

    def initParams(self, rng=None):
        #print "Linear: input_shape=", input_shape
        if rng == None: rng = np.random.RandomState()
        W_shape = (self.NU, self.NV, self.Nout)
        self.W = rng.normal(size=W_shape, scale=1.0/math.sqrt(self.NU + self.NV))
        
    def fprop(self, x, state_in):
        u, v = x
        y = np.einsum("bi,bj,ijk->bk", u, v, self.W)
        return y, None
        
    def bprop(self, x, state_in, y, state_out, gy, gState):
        u, v = x
        n_mb = len(u)
        assert len(v) == n_mb
        gu = np.einsum("bi,bk,jki->bj", gy, v, self.W)
        gv = np.einsum("bi,bj,jki->bk", gy, u, self.W)
        gw = np.einsum("bk,bi,bj->ijk", gy, u, v)
        return [gu, gv], None, [gw/n_mb]
        
    def regularize(self):
        self.W *= (1.0-self.WeightDecay)

    def getParams(self):
        return (self.W,)

    def setParams(self, tup):
        self.W, = tup

    def getParamsAsDict(self):     
        return {"W": self.W.copy()}
        
    def setParamsAsDict(self, dct):      
        self.W = dct["W"].copy()
        
        
