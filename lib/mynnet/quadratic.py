import numpy as np
from layers import Layer
import math

class Quadratic(Layer):
  
    def __init__(self, inp, out_size, weight_decay=1.e-5,
                name = None, applier = None):
        Layer.__init__(self, [inp], name)
        # input layer is expected to have output shape (sequence_length, in_size)
        self.Nin = inp.shape[0]
        self.Nout = out_size
        self.OutShape = (self.Nout,)
        self.WeightDecay = weight_decay

    def initParams(self, rng=None):
        #print "Linear: input_shape=", input_shape
        if rng == None: rng = np.random.RandomState()
        W_shape = (self.Nin, self.Nin, self.Nout)
        self.W = rng.normal(size=W_shape, scale=1.0/math.sqrt(self.Nin*self.Nin))
        self.W += self.W.transpose((1,0,2))
        self.W /= math.sqrt(2.0)
        
    def fprop(self, x, state_in):
        x = x[0]
        return np.einsum("bi,bj,ijk->bk", x, x, self.W), None
        
    def bprop(self, x, state_in, y, state_out, gy, gState):
        x = x[0]
        n_mb = len(x)
        #print "gY:", gY.shape, "   x:", self.X.shape
        gW = np.einsum("bi,bj,bk->ijk", x, x, gy)/2
        xw = x.dot(self.W+self.W.transpose(1,0,2))         #[b,in] dot [in,in,out] -> [b, in, out]
        gx = np.einsum("bj,bij->bi", gy, xw)
        return [gx], None, [gW/n_mb]
        
    def regularize(self):
        self.W *= (1.0-self.WeightDecay)

    def getParams(self):
        return (self.W,)
        
    def setParams(self, tup):
        self.W = tup[0]

    def getParamsAsDict(self):
        return {"W":self.W}
        
    def setParamsAsDict(self, dct):
        self.W = dct["W"]
        
        
