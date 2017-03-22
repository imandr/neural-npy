import numpy as np
from mynnet import Layer
from cconv import pool, pool_back


class Vote(Layer):
    def __init__(self, inputs, mode='max', name=None):
        Layer.__init__(self, inputs, name)
        s = inputs[0].shape
        #print s
        assert len(s) == 1, "Vote accepts only 1-dimensional inputs for now"
        for inp in inputs:
            assert inp.shape == s, "Shapes of all inputs of the Vote layer must be the same"
        self.Mode = mode
        self.OutShape = s
        
    def fprop(self, x, in_state):
        # x: [ninp, mbatch, n]
        ninp = len(x)
        mb, n = x[0].shape
        x = np.array(x).transpose((1,0,2)).reshape((mb,ninp,1,n))      # [mbatch, ninp, 1, n]
        y, pool_index = pool(x, ninp, 1)        # y, pool_index: [mbatch, 1, 1, n]
        y = y.reshape((mb, n))
        #print "fprop: y:", y.shape, "    index:",pool_index.shape
        return y, pool_index
        
    def bprop(self, x, state_in, y, state_out, gy, gState):
        # gy: [mb, n]
        ninp = len(x)
        mb, n = gy.shape
        gy = gy.reshape((mb, 1, 1, n))
        pool_index = state_out      # [mb, 1, 1, n]
        gx = pool_back(gy, pool_index, ninp, 1, ninp, 1)        # gx -> [mb, ninp, 1, n]
        gx = gx.reshape((mb, ninp, n)).transpose((1,0,2))
        #print "bprop: gx:", gx.shape
        return gx, None, []
        

if __name__ == '__main__':
    from mynnet import Model, InputLayer, L2Regression, Linear, Tanh
    
    np.set_printoptions(precision=3)
    
    in1 = InputLayer((7,))
    in2 = InputLayer((7,))
    
    v = Vote((in1, in2))
    #l = Tanh(Linear(v, 10))
    
    #m = Model([in1, in2, in3], v, L2Regression(v))
    #m.check_gradients(mb_size=3)
    
    mb = 2
    n = 5
    
    x1 = np.random.random((mb, n))
    x2 = np.random.random((mb, n))
    
    y, state_out = v.fprop([x1, x2], None)
    
    print "x1:", x1
    print "x2:", x2
    print "y: ", y
        
    gy = np.random.random((mb, n))
    
    print "gy:", gy
    
    gx, gstate, gparams = v.bprop([x1, x2], None, y, state_out, gy, None)
    
    print "gx:", gx
    
    l = Tanh(Linear(v, 5))
    m = Model([in1, in2], l, L2Regression(l))
    m.check_gradients(mb_size=53)
    