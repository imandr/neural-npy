import numpy as np
from cconv import convolve, pool, pool_back
import scipy.signal as sig
import math, time
from layers import Layer, ParamMixin

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
    
class Conv(ParamMixin, Layer):
    def __init__(self, inp, filter_x, filter_y, out_channels, name=None, 
                applier = None, weight_decay=1.0e-5):
        # filter_xy_shape is (fh, fw) 
        Layer.__init__(self, inp, name)
        ParamMixin.__init__(self, applier)
        #print self.InShape
        assert len(self.InShape) == 3
        self.filter_xy_shape = (filter_x, filter_y)
        self.out_channels = out_channels
        self.in_channels = self.InShape[2]
        self.filter_shape = self.filter_xy_shape + (self.in_channels, self.out_channels)   
             # (rows, columns, channels_in, channels_out)
        self.weight_decay = weight_decay
        self.OutImageShape = self.out_shape()
        self.gW = None
        

    def init_layer(self, rng):
        nin = np.prod(self.filter_xy_shape[:3])   # x*y*in_channels
        self.W = np.asarray(rng.normal(size=self.filter_shape, scale=1.0/math.sqrt(nin)), dtype=np.float32)
        #print self.W.dtype
        self.b = np.zeros(self.out_channels, dtype=np.float32)

    def fprop(self, inp):
        #print "conv.fprop"
        self.X = inp
        out_image_shape = self.output_shape(inp[0].shape)
        out_shape = (inp.shape[0],) + out_image_shape
        
        #print y.shape
        self.Y = convolve_xw(inp, self.W, 'valid') + self.b
        #print self.Y.shape
        return self.Y

    def reset_grads(self):
        #print "conv.reset_grads"
        self.gW = None
        self.gb = None
        self.gX = None
        self.gY = None

    def bprop(self, gY):
        #print "conv.bprop"
        n_imgs = gY.shape[0]

        gW = convolve_xy(self.X, gY)/n_imgs
        #print gW.shape

        w_flip = self.W[::-1,::-1,:,:]
        w_flip = np.transpose(w_flip, (0,1,3,2))
        gX = convolve_xw(gY, w_flip, 'full')

        gb = np.sum(gY, axis=(0, 1, 2))/n_imgs
        
        if self.gW is None:
            self.gW = gW
            self.gb = gb
            self.gX = gX
            self.gY = gY.copy()
        else:
            self.gW += gW
            self.gb += gb
            self.gX += gX
            self.gY += gY        
        return gX

    def addDeltas(self, dw, db):
        self.W += dw
        self.b += db
        
    def regularize(self):
        self.W *= (1.0-self.weight_decay)

    def __get_params(self):
        return self.W, self.b

    def __set_params(self, tup):
        self.W, self.b = tup

    params = property(__get_params, __set_params)
    
    def param_grads(self):
        return self.gW, self.gb

    def param_incs(self):
        return self.dW, self.db
        
    def dump_params(self, prefix):
        return {
            prefix+"/W":    self.W,
            prefix+"/B":    self.b
        }

    def restore_params(self, prefix, dct):
        self.W = dct[prefix+"/W"]
        self.b = dct[prefix+"/B"]

    def output_shape(self, input_shape):
        #print "inp_shape:", input_shape
        h = input_shape[0]-self.filter_shape[0]+1
        w = input_shape[1]-self.filter_shape[1]+1
        return (h, w, self.filter_shape[3])

class Pool_(Layer):
    def __init__(self, inp, pool_shape, mode='max', name=None):
        Layer.__init__(self, inp, name)
        self.Mode = mode
        self.pool_h, self.pool_w = pool_shape
        
    def fprop(self, inp):
        self.X = inp
        pool = [
            [
                img[iy::self.pool_h, ix::self.pool_w, :]
                    for iy in range(self.pool_h)
                        for ix in range(self.pool_w)
            ]
            for img in inp     # this loop is over the minibatch
        ]       

        pool = np.array(pool)       # [nb, px*py, nx/px, ny/py, nc]
        #print "pool: ", pool.shape

        pool_inx = np.argmax(pool, axis=1)  # [nb, nx/px, ny/py, nc], values 0...px*py-1
        self.pool_inx_dx = pool_inx % self.pool_w
        self.pool_inx_dy = pool_inx / self.pool_w
        self.Y = np.max(pool, axis=1)       # [nb, nx/px, ny/py, nc]
        return self.Y

    def bprop(self, gy):
        self.gY = gy
        nb, py, px, nc = gy.shape

        gy_t = gy.flat
        pool_inx_dx_t = self.pool_inx_dx.flat
        pool_inx_dy_t = self.pool_inx_dy.flat
        gx = np.zeros(self.X.shape)

        for k, g in enumerate(gy_t):
            ib, iy, ix, ic = np.unravel_index(k, gy.shape)
            off_x = pool_inx_dx_t[k]
            off_y = pool_inx_dy_t[k]
            gx[ib, iy*self.pool_h+off_y, ix*self.pool_w+off_x, ic] = g
        self.gX = gx
        return gx

    def output_shape(self, input_shape):
        return (input_shape[0]//self.pool_h,
                 input_shape[1]//self.pool_w,
                 input_shape[2]
                 )

class Pool(Layer):
    def __init__(self, inp, pool_shape, mode='max', name=None):
        Layer.__init__(self, inp, name)
        self.Mode = mode
        self.pool_h, self.pool_w = pool_shape
        
    def fprop(self, inp):
        self.X = inp
        #print "x:", self.X.dtype, self.X
        self.Y, self.PoolIndex = pool(inp, self.pool_h, self.pool_w)
        #print "y:", self.Y
        #print "index:", self.PoolIndex
        return self.Y

    def bprop(self, gy):
        self.gY = gy
        self.gX = pool_back(gy, self.PoolIndex, self.pool_h, self.pool_w, 
            self.X.shape[1], self.X.shape[2])
        #print "gx:", self.gX.dtype, self.gX
        return self.gX

    def output_shape(self, input_shape):
        return ((input_shape[0]+self.pool_h-1)//self.pool_h,
                 (input_shape[1]+self.pool_w-1)//self.pool_w,
                 input_shape[2]
                 )

