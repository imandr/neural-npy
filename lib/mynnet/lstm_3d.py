"""
This is a batched LSTM forward and backward pass
"""
import numpy as np
from mynnet import Layer, InputLayer, LSTM, LastRow
from .appliers import AdaDeltaApplier

class LSTM_3D(Layer):
  
    def __init__(self, inp, nxy, nout,
                reversed = False, 
                weight_decay=1.e-5,
                name = None, applier = None):
        Layer.__init__(self, [inp], name)
        # input layer is expected to have shape (nz, ny, nx)
        
        assert len(inp.shape) == 3
        nz, ny, nx = inp.shape
        self.NX = nx
        self.NXY = nxy
        self.NOut = nout
        self.Inp2D = InputLayer((ny, nx))          # (ny, nx)
        self.LSTM_XY = LSTM(self.Inp2D, self.NXY, 
            last_row_only = True, weight_decay=weight_decay, applier=AdaDeltaApplier())
        self.InpZ = InputLayer((nz, self.NXY))
        self.LSTM_Z = LSTM(self.InpZ, self.NOut,
            last_row_only = True, weight_decay=weight_decay, applier=AdaDeltaApplier())
        self.OutShape = self.LSTM_Z.shape
        
        #print "LSTM_3D.__init__: XY.shape:%s, Z.shape:%s" % (self.LSTM_XY.shape, self.LSTM_Z.shape)
        #print "                : XY.NC:%s, Z.NC:%s" % (self.LSTM_XY.NC, self.LSTM_Z.NC)
        
    
    def initParams(self, rng = None):
        # in_shape is (sequence, nin)
        if rng == None: rng = np.random.RandomState()
        self.LSTM_XY.initParams(rng)
        self.LSTM_Z.initParams(rng)
        
    def fprop(self, xyz, in_state):
        xyz = xyz[0]
        mb, nz, ny, nx = xyz.shape
        assert nx == self.NX
        XYOut = np.empty((mb, nz, self.NXY))
        XYStateOut = []
        XYCaches = []
        for iz in xrange(nz):
            xy = xyz[:,iz,:,:]
            out, state_out = self.LSTM_XY.fprop([xy], None)
            XYOut[:,iz,:] = out
            XYStateOut.append(state_out)
            XYCaches.append(self.LSTM_XY.Cache)
        self.XYCaches = XYCaches
        self.XYOut = XYOut
        #print "LSTM_3D.fprop: XYOut:", XYOut.shape
        self.XYStateOut = XYStateOut
        zout, out_state = self.LSTM_Z.fprop([XYOut], in_state)
        #print "fprop: zout:", zout.shape
        return zout, out_state
        
    def bprop(self, xyz, state_in, y, state_out, gy, gState):
        # xyx: [mb, nz, nx, ny]
        # y: [mb, NOut]
        #
        xyz = xyz[0]
        mb, nz, ny, nx = xyz.shape
        assert nx == self.NX
        #print "LSTM_Z.bprop"
        gxy, gxy_state, gw_z = self.LSTM_Z.bprop([self.XYOut], state_in, y, state_out, gy, gState)
        gxy = gxy[0]
        gw_z = gw_z[0]
        #print "bprop: gxy:", gxy.shape
        gxyz = np.empty_like(xyz)
        #print "gxyz:", gxyz.shape
        gw_xy = None
        #print "xyz:", xyz.shape
        #print "self.XYOut:", self.XYOut.shape
        for iz in xrange(nz):
            #print "LSTM_Z.bprop(%d)" % (iz,)
            self.LSTM_XY.Cache = self.XYCaches[iz]
            gxyz_z, _, gwxy_z = self.LSTM_XY.bprop([xyz[:,iz,:,:]], None, self.XYOut[:,iz,:], 
                    self.XYStateOut[iz], gxy[:,iz,:], None)    
            gxyz_z = gxyz_z[0]
            gwxy_z = gwxy_z[0]
            if gw_xy is None:
                gw_xy = gwxy_z.copy()
            else:
                gw_xy += gwxy_z
            #print "gxy:", gxy.shape
            gxyz[:,iz,:,:] = gxyz_z
        return [gxyz], None, [gw_xy, gw_z]
#
# "Canonic" inerface: x and y are shaped as (batch, sequence, data)
#

    def regularize(self):
        self.LSTM_Z.regularize()
        self.LSTM_XY.regularize()

    def getParams(self):
        return self.LSTM_XY.getParams() + self.LSTM_Z.getParams()
        
    def setParams(self, tup):
        self.LSTM_XY.setParams(tup[0])
        self.LSTM_Z.setParams(tup[1])

    def getParamsAsDict(self):
        raise NotImplementedError
        
    def setParamsAsDict(self, dct):
        raise NotImplementedError
        
if __name__ == '__main__':
    inp = InputLayer((3,5,7))
    lstm_3d = LSTM_3D(inp, 11, 13)
    lstm_3d.initParams()
    lstm_3d.checkGradients(mb=17)
    
    
    