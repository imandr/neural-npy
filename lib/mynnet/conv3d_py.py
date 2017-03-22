import numpy as np
import math, time
from mynnet import Layer

class Conv3D(Layer):
    def __init__(self, inp, filter_x, filter_y, filter_z, out_channels, 
                strides = (1,1,1),  name=None, 
                applier = None, weight_decay=1.0e-5):
        # filter_xy_shape is (fh, fw) 
        Layer.__init__(self, [inp], name)
        #print self.InShape
        assert len(inp.shape) == 4      # x, y, z, channels
        self.filter_xyz_shape = (filter_x, filter_y, filter_z)
        self.FX, self.FY, self.FZ = filter_x, filter_y, filter_z
        self.out_channels = out_channels
        self.in_channels = inp.shape[3]
        self.filter_shape = self.filter_xyz_shape + (self.in_channels, self.out_channels)   
             # (rows, columns, channels_in, channels_out)
        self.weight_decay = weight_decay
        self.Strides = strides
        self.OutShape = (
                (inp.shape[0]-filter_x+strides[0])/strides[0], 
                (inp.shape[1]-filter_y+strides[1])/strides[1], 
                (inp.shape[2]-filter_z+strides[2])/strides[2], 
                out_channels
            )
        
    def initParams(self, rng=None):
        if rng is None: rng = np.random.RandomState()
        nin = np.prod(self.filter_xyz_shape[:4])   # x*y*z*in_channels
        self.W = np.asarray(rng.normal(size=self.filter_shape, scale=1.0/math.sqrt(nin)), dtype=np.float32)
        #print self.W.dtype
        self.b = np.zeros(self.out_channels, dtype=np.float32)

    def fprop(self, x, in_state):
        x = x[0]
        WFlat = self.W.reshape(-1, self.out_channels)
        nb = len(x)
        y = np.empty((nb,)+self.OutShape, dtype=x.dtype)
        for ib in xrange(nb):
            for ix in xrange(self.OutShape[0]):
                for iy in xrange(self.OutShape[1]):
                    for iz in xrange(self.OutShape[2]):
                        patch = x[ib,
                            ix*self.Strides[0]:ix*self.Strides[0]+self.FX,
                            iy*self.Strides[1]:iy*self.Strides[1]+self.FY,
                            iz*self.Strides[2]:iz*self.Strides[2]+self.FZ,
                            :
                        ].reshape((-1,))
                        y[ib, ix, iy, iz] = patch.dot(WFlat)
        y += self.b
        return y, None

    def bprop(self, x, state_in, y, state_out, gY, gState):
        x = x[0]
        #print "conv.bprop"
        nb = x.shape[0]

        gw = np.zeros((self.FX*self.FY*self.FZ*self.in_channels, self.out_channels))
        gx = np.zeros_like(x)
        w_flip = self.W[::-1,::-1,::-1,:,:]
        for ib in xrange(nb):
            for ix in xrange(self.OutShape[0]):
                ix0 = ix*self.Strides[0]
                for iy in xrange(self.OutShape[1]):
                    iy0 = iy*self.Strides[1]
                    for iz in xrange(self.OutShape[2]):
                        iz0 = iz*self.Strides[2]
                        patch = x[ib,
                            ix0:ix0+self.FX,
                            iy0:iy0+self.FY,
                            iz0:iz0+self.FZ,:]      # [fx, fy, fz, cin]
                        #print patch.shape, gY[ib, ix, iy, iz, :].shape
                        gw += np.outer(patch, gY[ib, ix, iy, iz, :])
                        gx[ib,
                            ix0:ix0+self.FX,
                            iy0:iy0+self.FY,
                            iz0:iz0+self.FZ,:] += self.W.dot(gY[ib,ix,iy,iz,:]) # [fx,fy,fz,cin,cout].[cout]
        gb = np.sum(gY, axis=(0,1,2,3))/nb
        gw = gw.reshape(self.W.shape)/nb
        return [gx], None, (gw, gb)

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

class Pool3D(Layer):
    def __init__(self, inp, pool_shape, mode='max', name=None):
        Layer.__init__(self, [inp], name)
        self.Mode = mode
        self.pool_x, self.pool_y, self.pool_z = pool_shape
        self.OutShape = self.output_shape(inp.shape)
        print "Pool3D: input shape:", inp.shape, "  out shape:", self.OutShape
        
    def fprop(self, x, in_state):
        #print "x:", self.X.dtype, self.X
        x = x[0]
        nb = len(x)
        y_shape = (nb,)+self.output_shape(x.shape[1:])
        out = np.empty(y_shape)
        inx_shape = y_shape + (3,)
        pool_index = np.empty(inx_shape, dtype=np.int8)
        #print "pool_index shape:", pool_index.shape
        for ib in xrange(nb):
            for ix in xrange(y_shape[1]):
                ix0 = ix * self.pool_x
                ix1 = min(x.shape[1], ix0 + self.pool_x)
                for iy in xrange(y_shape[2]):
                    iy0 = iy * self.pool_y
                    iy1 = min(x.shape[2], iy0 + self.pool_y)
                    for iz in xrange(y_shape[3]):
                        iz0 = iz * self.pool_z
                        iz1 = min(x.shape[3], iz0 + self.pool_z)
                        patch = x[ib, ix0:ix1, iy0:iy1, iz0:iz1, :]
                        patch_flat = patch.reshape((-1, patch.shape[-1]))
                        out[ib,ix,iy,iz,:] = np.max(patch_flat, axis=0)
                        off = np.argmax(patch_flat, axis=0)
                        inx = np.array(np.unravel_index(off, patch.shape[:-1])).T
                        #print off, inx
                        #print pool_index[ib, ix, iy, iz, :, :].shape, inx.shape
                        pool_index[ib, ix, iy, iz, :, :] = inx
        return out, pool_index

    def bprop(self, x, state_in, y, state_out, gy, gState):
        x = x[0]
        pool_index = state_out
        
        gx = np.zeros_like(x)
        nb = len(x)
        y_shape = y.shape
        for ib in xrange(nb):
            for ix in xrange(y_shape[1]):
                ix0 = ix * self.pool_x
                for iy in xrange(y_shape[2]):
                    iy0 = iy * self.pool_y
                    for iz in xrange(y_shape[3]):
                        iz0 = iz * self.pool_z
                        for c in xrange(y_shape[4]):
                            offx, offy, offz = tuple(pool_index[ib, ix, iy, iz, c, :])
                            #print "bprop:",ib, ix, iy, iz, " off:", off, offx, offy, offz
                            gx[ib, ix0+offx, iy0+offy, iz0+offz, c] = gy[ib, ix, iy, iz, c]        
        return [gx], None, []

    def output_shape(self, input_shape):
        return ((input_shape[0]+self.pool_x-1)//self.pool_x,
                 (input_shape[1]+self.pool_y-1)//self.pool_y,
                 (input_shape[2]+self.pool_z-1)//self.pool_z,
                 input_shape[3]
                 )


if __name__ == "__main__":
    from mynnet import InputLayer, Model, L2Regression, ReLU, Flatten, Tanh, Linear, Model
    
    inp = InputLayer((11,13,17,3))
    c3d = ReLU(
        Pool3D(
            Conv3D(inp, 3,3,3, 7, strides=(2,2,2)),
            (2,2,2)
        )
    )
    l = Tanh(Linear(Flatten(c3d), 5))
    m = Model([inp], l, L2Regression(l))
    m.checkGradients()
