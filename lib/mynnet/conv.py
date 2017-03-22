import numpy as np
from cconv import convolve, pool, pool_back
import scipy.signal as sig
import math, time
from mynnet import Layer

def convolve_xw(inp, w, mode):
    # inp: (nb, nx, ny, nc_in)
    # w: (nx, ny, nc_in, nc_out)
    # returns (nb, x, y, nc_out)
    
    mode = 0 if mode == 'valid' else 1
    
    inp = inp.transpose((0,3,1,2))
    w = w.transpose((3,2,0,1))
    return convolve(inp, w, mode).transpose((0,2,3,1))
    
def convolve_xy(x, y):
    # x: (nb, nx, ny, nc_in)
    # y: (nb, mx, my, nc_out)       (mx,my) < (nx,ny)
    # returns (fx, fy, nc_in, nc_out)
    
    x = x.transpose((3,0,1,2))
    y = y.transpose((3,0,1,2))
    return convolve(x, y, 0).transpose((2,3,0,1))
    
class Conv(Layer):
    def __init__(self, inp, filter_x, filter_y, out_channels, name=None, 
                applier = None, weight_decay=1.0e-5):
        # filter_xy_shape is (fh, fw) 
        Layer.__init__(self, [inp], name)
        #print self.InShape
        assert len(inp.shape) == 3
        self.filter_xy_shape = (filter_x, filter_y)
        self.out_channels = out_channels
        self.in_channels = inp.shape[2]
        self.filter_shape = self.filter_xy_shape + (self.in_channels, self.out_channels)   
             # (rows, columns, channels_in, channels_out)
        self.weight_decay = weight_decay
        
        self.OutShape = (inp.shape[0]-filter_x+1, inp.shape[1]-filter_y+1, out_channels)
        

    def initParams(self, rng):
        nin = np.prod(self.filter_shape[:3])   # x*y*in_channels
        self.W = np.asarray(rng.normal(size=self.filter_shape, scale=1.0/math.sqrt(nin)), dtype=np.float32)
        #print self.W.dtype
        self.b = np.zeros(self.out_channels, dtype=np.float32)

    def fprop(self, x, in_state):
        y = convolve_xw(x[0], self.W, 'valid') + self.b
        #print self.Y.shape
        return y, None

    def bprop(self, x, state_in, y, state_out, gY, gState):
        x = x[0]
        #print "conv.bprop"
        n_imgs = x.shape[0]

        #print x.shape, gY.shape

        gW = convolve_xy(x, gY)
        #print gW.shape

        w_flip = self.W[::-1,::-1,:,:]
        w_flip = np.transpose(w_flip, (0,1,3,2))
        gx = convolve_xw(gY, w_flip, 'full')

        gb = np.sum(gY, axis=(0, 1, 2))
        return [gx], None, (gW/n_imgs, gb/n_imgs)

    def regularize(self):
        self.W *= (1.0-self.weight_decay)

    def getParams(self):
        return self.W, self.b

    def setParams(self, tup):
        self.W, self.b = tup
        
    def getParamsAsDict(self):
        return { 'W':self.W, 'b':self.b }

    def setParams(self, dct):
        self.W, self.b = dct['W'], dct['b']


class Pool(Layer):
    def __init__(self, inp, pool_shape, mode='max', name=None):
        Layer.__init__(self, [inp], name)
        self.Mode = mode
        self.pool_h, self.pool_w = pool_shape
        self.OutShape = self.output_shape(inp.shape)
        
    def fprop(self, x, in_state):
        #print "x:", self.X.dtype, self.X
        y, pool_index = pool(x[0], self.pool_h, self.pool_w)
        return y, pool_index

    def bprop(self, x, state_in, y, state_out, gy, gState):
        x = x[0]
        pool_index = state_out
        gx = pool_back(gy, pool_index, self.pool_h, self.pool_w, 
            x.shape[1], x.shape[2])
        return [gx], None, []

    def output_shape(self, input_shape):
        return ((input_shape[0]+self.pool_h-1)//self.pool_h,
                 (input_shape[1]+self.pool_w-1)//self.pool_w,
                 input_shape[2]
                 )

