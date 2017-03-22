from lstm_core import LSTM_Core
from mynnet import Layer
import numpy as np

class LSTM_Cored(Layer):
    def __init__(self, inp, out_size,
                reversed = False, 
                last_row_only = False, weight_decay=1.e-5,
                name = None, applier = None):
        Layer.__init__(self, [inp], name)
        
        # input layer is expected to have output shape (sequence_length, in_size)
        self.NC = out_size
        self.Nout = out_size
        self.SequenceLength, self.Nin = inp.shape
        self.Core = LSTM_Core(self.Nin, out_size)
        self.WeightDecay = weight_decay
        self.LastRowOnly = last_row_only
        self.Reversed = reversed                # ignored for now!
        if self.Reversed:   raise NotImplementedError
        self.OutShape = (self.Nout,) if self.LastRowOnly else (self.SequenceLength, self.Nout) 
        
    def initParams(self, rng=None):
        if rng is None: rng = np.random.RandomState()
        self.Core.initParams(rng)
        
    def fprop(self, x, in_state):
        x = x[0]
        #print "LSTM.fprop: x=", x.shape
        x = x.reshape((x.shape[0], -1, self.Nin))

        mb, n, nx = x.shape
        
        if in_state is None:    in_state = self.Core.zeroState(mb)
        
        state = in_state
        yout = np.empty((mb, n, self.Nout))
        context = []
        for t in xrange(n):
            y, state, ctx = self.Core.forward(x[:,t,:], state)
            yout[:,t,:] = y
            context.append(ctx)

        self.Context = context

        if self.LastRowOnly:
            yout = yout[:,-1,:]
            
        return yout, state
        
    def bprop(self, x,state_in, y, state_out, gy, gState):
        x = x[0]
        mb, n, nx = x.shape
        # if LastRowOnly == True, gy must be (batch, Nout)
        # otherwise (batch, sequence_length, Nout)
        if self.LastRowOnly:
            l = x.shape[1]     # sequence length
            gy_expanded = np.zeros((gy.shape[0], l, gy.shape[1]))
            gy_expanded[:,-1,:] = gy
            gy = gy_expanded
            
        gW = np.zeros_like(self.Core.W)
        gX = np.empty((mb, n, self.Nin))
        gstate = gState
        for t in reversed(xrange(n)):
            gx, gw, gstate = self.Core.backward(gy[:,t,:], gstate, self.Context[t])
            gX[:,t,:] = gx
            gW += gw
        
        return [gX], gstate, [gW/mb]

    def regularize(self):
        self.Core.W *= (1.0 - self.WeightDecay)

    def getParams(self):
        #print "MDRN_LSTM_2D.getParams: W=", self.LSTMC.W[-1,:10]
        return (self.Core.W,)
        
    def setParams(self, tup):
        self.Core.W = tup[0]

    def getParamsAsDict(self):
        raise NotImplementedError
        
    def setParamsAsDict(self, dct):
        raise NotImplementedError
