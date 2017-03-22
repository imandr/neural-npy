import numpy as np
from mynnet import Layer


NEEDS to be fixed for new Layer interface

class Distort(Layer):
  
    def __init__(self, inp, out_size, weight_decay=1.e-5,
                name = None, applier = None):
        Layer.__init__(self, [inp], name)
        # input layer is expected to have output shape (sequence_length, in_size)
        self.Nin = self.InShape[0]
        self.Nout = out_size
        self.Np = 2**self.Nin
        self.WeightDecay = weight_decay
        
            
    def init_layer(self, rng = None):
        if rng == None: rng = np.random.RandomState()
        self.Signs = np.empty((self.Nin, self.Np), dtype=np.int)        #[Nin, Np]
        
        for ip in range(self.Np):
            i = ip
            bits = []
            for j in range(self.Nin):
                bits.append(i%2)
                i /= 2
            self.Signs[:,ip] = np.array(bits[::-1])
        self.Signs *= 2
        self.Signs -= 1     # make it (-1,1)
        
        #self.W = rng.normal(size=(self.Np, self.Nout), scale=1.0)  # / np.sqrt(self.Np))
        self.W = rng.uniform(-2.0, 2.0, size=(self.Np, self.Nout))  # / np.sqrt(self.Np))
        #self.W = rng.choice((-2.0, 2.0), replace=True, size=(self.Np, self.Nout))  # / np.sqrt(self.Np))
        
        
    def fprop(self, X, state_in):
        X = X[0]
        nb = len(X)
        sx = np.outer(X, self.Signs).reshape((nb, self.Nin, self.Nin, self.Np)) # [Nb, Nin], [Nin, Np] -> [Nb, Nin, Nin, Np]
        sx = np.einsum("ijjk->ijk", sx)          # [Nb, Nin, Np]
        self.q = (sx + 1.0)/2                    # [Nb, Nin, Np]
        self.Q = np.prod(self.q, axis=1)         # [Nb, Nin, Np]-> [Nb, Np]
        y = self.Q.dot(self.W)                   # [Nb, Np] dot [Np, Nout] -> [Nb, Nout]
        return y, None
        
    def bprop(self, x, state_in, y, state_out, gy, gState):
        nb = gy.shape[0]
        D = np.empty((nb, self.Nin, self.Np), dtype=gy.dtype)       # [Nb, Nin, Np]
        for a in range(self.Nin):
            qq = self.q.copy()
            qq[:,a,:] = self.Signs[a,:]*0.5
            D[:,a,:] = np.prod(qq, axis=1)
        wd = D.dot(self.W)                          # (nb, nin, np) dot (np, nout) -> (nb, nin, nout)
        wd = wd.transpose((0,2,1))                  # -> (nb, nout, nin)
        gx = gy.dot(wd)                             # (nb, nout), (nb, nout, nin) -> (nb, nb, nin)
        gx = np.einsum("iij->ij", gx)          # -> (nb, nin)
        gw = self.Q.T.dot(gy)
        
        return gx, None, [gw/nb]
        
    def regularize(self):
        self.W *= (1.0 - self.WeightDecay)
        
#
# ParamMixin
#      
    def __get_params(self):
        return (self.W,)
        
    def __set_params(self, tup):
        self.W = tup[0]

    params = property(__get_params, __set_params)
    
    def param_incs(self):
        return (self.dW,)

    def param_grads(self):
        return (self.gW,)

    def dump_params(self, prefix):
        return {
            prefix+"/W":    self.W
        }

    def restore_params(self, prefix, dct):
        self.W = dct[prefix+"/W"]
      

class Distort_(ParamMixin, Layer):
  
    def __init__(self, inp, out_size, weight_decay=1.e-7,
                name = None, applier = None):
        Layer.__init__(self, inp, name)
        ParamMixin.__init__(self, applier)
        # input layer is expected to have output shape (sequence_length, in_size)
        self.Nin = self.InShape[0]
        self.Nout = out_size
        self.Np = 2**self.Nin
        self.WeightDecay = weight_decay
        
        self.Signs = np.empty((self.Nin, self.Np), dtype=np.int)        #[Nin, Np]
        
        for ip in range(self.Np):
            i = ip
            bits = []
            for j in range(self.Nin):
                bits.append(i%2)
                i /= 2
            self.Signs[:,ip] = np.array(bits[::-1])
        self.Signs *= 2
        self.Signs -= 1     # make it (-1,1)
            
    def init_layer(self, rng = None):
        if rng == None: rng = np.random.RandomState()
        self.W = rng.normal(size=(self.Np, self.Nout), scale= 3.0)  # / np.sqrt(self.Np))
        self.B = np.zeros(self.Nout)
        
        
    def fprop(self, X, state = None):
        # X is [Nb, Nin]
        self.X = X
        nb = len(X)
        sx = np.outer(X, self.Signs).reshape((nb, self.Nin, self.Nin, self.Np)) # [Nb, Nin], [Nin, Np] -> [Nb, Nin, Nin, Np]
        sx = np.einsum("ijjk->ijk", sx)          # [Nb, Nin, Np]
        self.q = (sx + 1.0)/2                    # [Nb, Nin, Np]
        self.Q = np.prod(self.q, axis=1)         # [Nb, Nin, Np]-> [Nb, Np]
        y = self.Q.dot(self.W)                   # [Nb, Np] dot [Np, Nout] -> [Nb, Nout]
        self.Y = y + self.B
        return self.Y
        
    def bprop(self, gy):
        nb = gy.shape[0]
        D = np.empty((nb, self.Nin, self.Np), dtype=gy.dtype)       # [Nb, Nin, Np]
        for a in range(self.Nin):
            qq = self.q.copy()
            qq[:,a,:] = self.Signs[a,:]*0.5
            D[:,a,:] = np.prod(qq, axis=1)
        wd = D.dot(self.W)                          # (nb, nin, np) dot (np, nout) -> (nb, nin, nout)
        wd = wd.transpose((0,2,1))                  # -> (nb, nout, nin)
        gx = gy.dot(wd)                             # (nb, nout), (nb, nout, nin) -> (nb, nb, nin)
        self.gx = np.einsum("iij->ij", gx)          # -> (nb, nin)
        
        self.gW = self.Q.T.dot(gy)/nb
        self.gB = np.mean(gy, axis=0)
        return self.gx
        
    def addDeltas(self, dW, dB):
        self.W += dW
        self.B += dB
        
    def regularize(self):
        self.W *= (1.0 - self.WeightDecay)
        
    def resetGrads(self):
        self.gW = np.zeros_like(self.W)
        self.gB = np.zeros_like(self.B)

    def output_shape(self, input_shape):
        return (self.Nout,)
#
# ParamMixin
#      
    def __get_params(self):
        return (self.W, self.B)
        
    def __set_params(self, tup):
        self.W, self.B = tup

    params = property(__get_params, __set_params)
    
    def param_incs(self):
        return (self.dW, self.dB)

    def param_grads(self):
        return (self.gW, self.gB)

    def dump_params(self, prefix):
        return {
            prefix+"/W":    self.W,
            prefix+"/B":    self.b
        }

    def restore_params(self, prefix, dct):
        self.W = dct[prefix+"/W"]
        self.b = dct[prefix+"/B"]

      

