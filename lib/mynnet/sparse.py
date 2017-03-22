import numpy as np
from layers import Layer, ParamMixin
import math

NEEDS to be fixed for new Layer interface

class Sparse(ParamMixin, Layer):
  
    def __init__(self, inp, out_size, weight_decay=1.e-5,
                name = None, applier = None):
        Layer.__init__(self, inp, name)
        ParamMixin.__init__(self, applier)
        # input layer is expected to have output shape (sequence_length, in_size)
        self.Nin = self.InShape[0]
        self.Nout = out_size
        self.WeightDecay = weight_decay

    def init_layer_(self, rng=None):
        #print "Linear: input_shape=", input_shape
        if rng == None: rng = np.random.RandomState()
        W_shape = (self.Nin, self.Nin, self.Nout)
        self.W = np.zeros(W_shape)
        self.Mask = np.zeros(W_shape)
        self.Diagonal2 = np.ones(W_shape)
        for k in range(self.Nout):
            for ii in range(self.Nin):
                self.W[ii,ii,k] = rng.normal(scale=1.0/math.sqrt(self.Nin*2))
                self.Mask[ii,ii,k] = 1.0
            n_off_d = self.Nin - 1 if self.Nin <= 2 else self.Nin
            for _ in range(n_off_d):
                done = False
                while not done:
                    i = rng.randint(1, self.Nin)
                    j = rng.randint(0, i)
                    done = self.Mask[i,j,k] == 0.0
                self.W[i,j,k] = rng.normal(scale=1.0/math.sqrt(self.Nin*2))
                self.Mask[i,j,k] = 1.0
        
    def init_layer(self, rng=None):
        #print "Linear: input_shape=", input_shape
        if rng == None: rng = np.random.RandomState()
        W_shape = (self.Nin, self.Nin, self.Nout)
        self.W = np.zeros(W_shape)
        self.Mask = np.zeros(W_shape)
        self.Diagonal2 = np.ones(W_shape)
        for k in range(self.Nout):
            for i in range(self.Nin):
                self.W[i,i,k] = rng.normal(scale=1.0/math.sqrt(self.Nin*2))
                self.Mask[i,i,k] = 1.0
            
            if self.Nin <= 2:
                self.W[0,1,k] = rng.normal(scale=1.0/math.sqrt(self.Nin*2))
                self.Mask[0,1,k] = 1.0
            else:
                for i in range(self.Nin):
                    j = (i+1) % self.Nin
                    self.W[i,j,k] = rng.normal(scale=1.0/math.sqrt(self.Nin*2))
                    self.Mask[i,j,k] = 1.0
        print "shape:", self.W.shape
        print "W[:,:,0]:", self.W[:,:,0]
        
    def fprop(self, x):
        self.X = x
        self.Y = np.einsum("bi,bj,ijk->bk", x, x, self.W)
        #print "fprop y:", self.Y.shape
        return self.Y
        
    def bprop(self, gY):
        self.gY = gY
        #print "gY:", gY.shape, "   x:", self.X.shape
        gW = np.einsum("bi,bj,bk->ijk", self.X, self.X, gY) * self.Mask/2
        xw = self.X.dot(self.W+self.W.transpose(1,0,2))         #[b,in] dot [in,in,out] -> [b, in, out]
        gX = np.einsum("bj,bij->bi", gY, xw)

        if self.gW is None:
            self.gW = gW
            self.gX = gX
            self.gY = gY.copy()
        else:
            self.gW += gW
            self.gX += gX
            self.gY += gY        
        return gX
        
    def reset_grads(self):
        self.gW = None
        self.gb = None
        self.gX = None
        self.gY = None

    def addDeltas(self, dw):
        self.W += dw
        
    def regularize(self):
        self.W *= (1.0-self.WeightDecay)

    def __get_params(self):
        return (self.W,)

    def __set_params(self, tup):
        self.W = tup[0]

    params = property(__get_params, __set_params)
    
    def param_incs(self):
        return (self.dW,)

    def param_grads(self):
        return (self.gW,)

    def output_shape(self, input_shape):
        return (self.Nout,)

    def dump_params(self, prefix):
        return {
            prefix+"/W":    self.W,
        }

    def restore_params(self, prefix, dct):
        self.W = dct[prefix+"/W"]
    
        
        
