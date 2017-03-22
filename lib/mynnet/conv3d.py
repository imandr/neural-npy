import numpy as np
from cconv import convolve_3d, convolve_back_3d, pool_3d, pool_back_3d
import scipy.signal as sig
import math, time
from mynnet import Layer

def optimize(a):
    strides = a.__array_interface__["strides"]
    if strides is None:
        return a

    if strides == sorted(strides, reverse=True) and strides[-1] == a.itemsize:
        return a

    return a.copy()
    
class Conv3D(Layer):
    def __init__(self, inp, filter_x, filter_y, filter_z, out_channels, 
                stride = 1,
                name=None, 
                applier = None, weight_decay=1.0e-5):
        # filter_xy_shape is (fh, fw) 
        Layer.__init__(self, [inp], name)
        #print self.InShape
        assert len(inp.shape) == 4
        self.filter_xyz_shape = (filter_x, filter_y, filter_z)
        self.out_channels = out_channels
        self.in_channels = inp.shape[3]
        self.filter_shape =  (self.out_channels, filter_x, filter_y, filter_z, self.in_channels)   
             # (rows, columns, channels_in, channels_out)
        self.weight_decay = weight_decay
        self.Stride = stride
        
        self.OutShape = self.outShape(inp.shape[:3]) + (out_channels,)

    def outShape(self, in_shape):
        return (
            (in_shape[0]-self.filter_xyz_shape[0]+self.Stride)/self.Stride,
            (in_shape[1]-self.filter_xyz_shape[1]+self.Stride)/self.Stride,
            (in_shape[2]-self.filter_xyz_shape[2]+self.Stride)/self.Stride
        )
        
        
        

    def initParams(self, rng = None):
        if rng == None: rng = np.random.RandomState()
        nin = np.prod(self.filter_shape[1:])   # x*y*z*in_channels
        self.W = rng.normal(size=self.filter_shape, scale=1.0/math.sqrt(nin))
        #print self.W.dtype
        self.b = np.zeros(self.out_channels)

    def fprop(self, x, in_state):
        x = x[0]        #optimize(x[0])
        #print "fprop:convolve_3d, x:%s, %x" % (x.shape, id(x))
        #print x
        y = convolve_3d(x, self.W, self.Stride, 1, self.outShape(x.shape[1:4])) + self.b
        return y, None

    def bprop(self, x, state_in, y, state_out, gY, gState):
        x = x[0]
        #print "conv.bprop"
        n_imgs = x.shape[0]
        
        xt = optimize(x.transpose((4,1,2,3,0)))
        gyt = optimize(gY.transpose((4,1,2,3,0)))
        sw = self.W.shape[1:4]
        #print "bprop:convolve_3d"
        gW = convolve_3d(xt, gyt, 1, self.Stride, sw).transpose((4,1,2,3,0))
        
        wt = optimize(self.W.transpose((4,1,2,3,0)))
        gyo = optimize(gY)
        #print "bprop:convolve_back_3d"
        gx = convolve_back_3d(gyo, wt, self.Stride, x.shape[1:4])
        
        gb = np.sum(gyo, axis=(0, 1, 2, 3))
        
        return [gx], None, (gW/n_imgs, gb/n_imgs)

    def regularize(self):
        self.W *= (1.0-self.weight_decay)

    def getParams(self):
        return self.W, self.b

    def setParams(self, tup):
        self.W, self.b = tup
        
    def getParamsAsDict(self):
        return { 'W':self.W, 'b':self.b }

    def setParamsAsDict(self, dct):
        self.W, self.b = dct['W'], dct['b']


class Pool3D(Layer):
    def __init__(self, inp, pool_shape, mode='max', name=None):
        Layer.__init__(self, [inp], name)
        self.Mode = mode
        self.pool_x, self.pool_y, self.pool_z = pool_shape
        self.OutShape = self.output_shape(inp.shape)
        
    def fprop(self, x, in_state):
        #print "x:", self.X.dtype, self.X
        #print "pool_3d"
        y, pool_index = pool_3d(x[0], self.pool_x, self.pool_y, self.pool_z)
        #print "pool_3d done"
        return y, pool_index

    def bprop(self, x, state_in, y, state_out, gy, gState):
        x = x[0]
        pool_index = state_out
        #print "pool_back_3d"
        gx = pool_back_3d(gy, pool_index, self.pool_x, self.pool_y, self.pool_z, 
            x.shape[1], x.shape[2], x.shape[3])
        return [gx], None, []

    def output_shape(self, input_shape):
        return ((input_shape[0]+self.pool_x-1)//self.pool_x,
                 (input_shape[1]+self.pool_y-1)//self.pool_y,
                 (input_shape[2]+self.pool_z-1)//self.pool_z,
                 input_shape[3]
                 )

if __name__ == '__main__':
    from mynnet import InputLayer, Model, L2Regression
    def check_grads():
        inp = InputLayer((100,100,100,1))
        c = Conv3D(inp, 5,5,5, 15, stride=3)
        p = Pool3D(c, (2,2,2))
        m = Model([inp], p, L2Regression(p))
        m.checkGradients()
        
    def check_conv():
        inp = InputLayer((5,5,5,1))
        c = Conv3D(inp, 2,2,2, 2, stride=1)
        c.initParams()
        c.W[...] = 1.0
        c.W[1,...] = -1.0
        x = np.ones((1,)+inp.shape)
        #x[...,1] = 0
        #print x
        print c.fprop([x], None)
        
    check_grads()
        
        
        