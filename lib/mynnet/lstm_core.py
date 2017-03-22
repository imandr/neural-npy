import numpy as np
import random

class LSTM_Core:
  
    def __init__(self, in_size, out_size):
        # input layer is expected to have output shape (sequence_length, in_size)
        hidden_size = out_size
        self.Nin = in_size
        self.NC = hidden_size
        self.Nout = out_size
        
    def initParams(self, rng = None):
        if rng is None: rng = np.random.RandomState()
        self.W = rng.normal(size=(self.Nin + self.NC + 1, 4 * self.NC),
                scale= 1.0 / np.sqrt(self.Nin+self.Nout))
        #print "WLSTM shape=", self.WLSTM.shape
        self.W[0,:] = 0 # initialize biases to zero
        self.W[0,self.NC:2*self.NC] = 2.0
        #self.W[0,:] = rng.normal(size=(4 * self.NC,), scale=2.0)
        
        
    def zeroState(self, mbsize):
        return np.zeros((2, mbsize, self.NC))
                
    def forward(self, X, state_in):
        # X is [mb, in_size]
        # state is [2, mb, out_size]    - 2 for H,C
        
        b,input_size = X.shape
        assert input_size == self.Nin, "input shape: %s,   Nin:%s" % (X.shape, self.Nin)
        d = self.NC # hidden size
        
        CIn, HIn = state_in[0], state_in[1]
        
        
        # Perform the LSTM forward pass with X as the input
        xphpb = self.W.shape[0] # x plus h plus bias, lol
        #print "Hout size:", Hout.shape
        IFOGf = np.empty((b, d * 4)) # after nonlinearity
        HXin = np.empty((b, xphpb)) # input [1, xt, ht-1] to each tick of the LSTM
        HXin[:,0] = 1 # bias
        HXin[:,1:input_size+1] = X
        HXin[:,input_size+1:] = HIn
        # compute all gate activations. dots: (most work is this line)
        
        # IFOG map:
        #   [:,0:d]     -- new C gate
        #   [:,d:2d]    -- old C gate (a.k.a. forget gate)
        #   [:,2d:3d]   -- output gate   (out = sigmoid(gate)*C)
        #   [:,3d:]     -- new C
        
        IFOG = HXin.dot(self.W)     
        # non-linearities
        IFOGf[:,:3*d] = 1.0/(1.0+np.exp(-IFOG[:,:3*d])) # sigmoids; these are the gates
        IFOGf[:,3*d:] = np.tanh(IFOG[:,3*d:]) # tanh
        # compute the cell activation
        
        
        Cout = IFOGf[:,:d] * IFOGf[:,3*d:] + IFOGf[:,d:2*d] * CIn        # Cout
        Ct = np.tanh(Cout)
        Hout = IFOGf[:,2*d:3*d] * Ct        # Hout
        y = Hout
        
        state_out = np.empty((2, b, self.NC))
        state_out[0] = Cout
        state_out[1] = Hout
        
        context = {}
        context['IFOGf'] = IFOGf
        context['IFOG'] = IFOG
        context['Ct'] = Ct
        context['HXin'] = HXin
        context['state_in'] = state_in          # Cin and Hin
        context['state_out'] = state_out        # includes Cout and Hout
        
        return y, state_out, context
        
    def backward(self, gY, gState_out, context): 
        IFOGf = context['IFOGf']
        IFOG = context['IFOG']
        CHout = context['state_out']
        Ct = context['Ct']
        HXin = context['HXin']
        CHin = context['state_in']

        gH = gY.copy()

        #C = CHout[0]
        #Hout = CHout[1]
        Cin = CHin[0]

        b = len(gY)
        d = self.NC

        input_size = self.Nin

        gIFOG = np.empty(IFOG.shape)        # was np.zeros
        gIFOGf = np.empty(IFOGf.shape)      # was np.zeros
        
        if not gState_out is None: 
            gC = gState_out[0].copy()           # carry over gradients from later
            gH += gState_out[1]
        else:
            gC = np.zeros(Cin.shape)
            
            
        # backprop tanh non-linearity first then continue backprop
        gC += (1-Ct**2) * (IFOGf[:,2*d:3*d] * gH)
    
        gIFOGf[:,:d] = IFOGf[:,3*d:] * gC
        gIFOGf[:,d:2*d] = Cin * gC
        gIFOGf[:,2*d:3*d] = Ct * gH     # [nb, Nc] * [nb, Nc] -> [nb, Nc]
        gIFOGf[:,3*d:] = IFOGf[:,:d] * gC
    
        # backprop activation functions
        gIFOG[:,3*d:] = (1 - IFOGf[:,3*d:] ** 2) * gIFOGf[:,3*d:]
        y = IFOGf[:,:3*d]
        gIFOG[:,:3*d] = (y*(1.0-y)) * gIFOGf[:,:3*d]
    
        # backprop matrix multiply
        gW = np.dot(HXin.transpose(), gIFOG)
        gHXin = gIFOG.dot(self.W.T)
    
        gX = gHXin[:,1:input_size+1]
        gState_in = np.empty((2, b, d))
        gState_in[0] = IFOGf[:,d:2*d] * gC
        gState_in[1] = gHXin[:,input_size+1:]

        return gX, gW, gState_in

    
    
if __name__ == '__main__':
    
    nx = 3
    nh = 5
    
    lstm = LSTM_Core(nx, nh)
    lstm.initParams()

    def L(y):
        return np.sum(y**2)/2

    def gL(y):
        return y
        
    def Lx(x, state_in, cell):
        nb = x.shape[0]
        y1, state_mid, b = cell.forward(x[:,:nx], state_in)
        y2, state_out, b = cell.forward(x[:,nx:], state_mid)
        y = np.empty((nb, nh*2))
        y[:,:nh] = y1
        y[:,nh:] = y2
        return L(y)
        
    def cmp_grads(func, arg, x, sin, p, gp, delta, N):
        pflat = p.flat
        ilst = range(len(pflat))
        if not N is None and len(ilst) > N:
            ilst = random.sample(ilst, N)
            ilst.sort()
            
        passed = True
        for i in ilst:
            psave = pflat[i]
            pflat[i] = psave - delta
            l1, _, _, _, _, _ = func(x, sin, arg)
            pflat[i] = psave + delta
            l2, _, _, _, _, _ = func(x, sin, arg)
            pflat[i] = psave
        
            g_c = (l2-l1)/(delta*2)
            g_a = gp.flat[i]
        
            flag = False
        
            if g_a == 0.0:
                if g_c == 0.0:
                    flag = not np.allclose(g_a, g_c)
                else:
                    flag = not np.allclose(g_a, g_c) and not np.allclose(g_a/g_c, 1.0)
            else:
                flag = not np.allclose(g_a, g_c) and not np.allclose(g_c/g_a, 1.0)
            if True or flag:
                print i, g_c, g_a, g_c-g_a
            passed = passed and not flag
        return passed

    def step2(cell):
        nb = 3
        nx = cell.Nin
        nh = cell.NC
    
        def Lx(x, state_in, cell):
            nb = len(x)
            nx = cell.Nin
            x1 = x[:,:nx].copy()
            x2 = x[:,nx:].copy()
            #x1[:,0] = 0
            y1, state_mid, context_1 = cell.forward(x1, state_in)
            y2, state_out, context_2 = cell.forward(x2, state_mid)
            ny = cell.Nout
            y = np.empty((nb, ny*2))
            y[:,:ny] = y1
            y[:,ny:] = y2
            return L(state_out), y, state_out, (context_1, context_2), \
                np.zeros_like(y), gL(state_out)

        x = np.random.random((nb, nx*2))
        state_in = np.random.random(cell.zeroState(nb).shape)
        l, y, state_out, (context_1, context_2), gY, gState = Lx(x, state_in, cell)
        
        gX2, gW2, gState_mid = lstm.backward(gY[:,nh:], gState, context_2)
        gX1, gW1, gState_in = lstm.backward(gY[:,:nh], gState_mid, context_1)
        gW = gW1 + gW2
        gX = np.concatenate((gX1, gX2), axis=1)
    
        N = 20
        delta = 0.0001

        print "gX ------", "OK" if cmp_grads(Lx, cell, x, state_in, x, gX, delta, N) else "failed"
        print "gW ------", "OK" if cmp_grads(Lx, cell, x, state_in, cell.W, gW, delta, N) else "failed"
        print "gState --", "OK" if cmp_grads(Lx, cell, x, state_in, state_in, gState_in, delta, N) else "failed"
        
    def single(cell):
        
        nx = cell.Nin
        nh = cell.NC
        nb = 3
        
    
        def Lx(x, state_in, cell):
            y, state_out, context = cell.forward(x, state_in)
            return L(state_out)+L(y), y, state_out, context
            
        x = np.random.random((nb, nx))
        state_in = np.random.random(cell.zeroState(nb).shape)
        l, y, state_out, context = Lx(x, state_in, cell)
        
        gly = gL(y)
        gls = gL(state_out)
            
        gX, gW, gState_in = lstm.backward(gly, gls, context)
    
        N = 20
        delta = 0.0001

        print "gX ------", "OK" if cmp_grads(Lx, cell, x, state_in, x, gX, delta, N) else "failed"
        print "gW ------", "OK" if cmp_grads(Lx, cell, x, state_in, cell.W, gW, delta, N) else "failed"
        print "gState --", "OK" if cmp_grads(Lx, cell, x, state_in, state_in, gState_in, delta, N) else "failed"
        
    def rms():
        import math
        nx = 100
        nh = 100
        nb = 1000
        
        rng = np.random.RandomState()

        lstm = LSTM_Core(nx, nh)
        lstm.initParams(rng)
        
        x = rng.normal(size=(nb, nx), scale= 1.0)
        in_state = rng.normal(size=(nb, nh), scale= 0.0001)
        y, out_state, context = lstm.forward(x, in_state)
        print y.shape
        print math.sqrt(np.mean(y*y))
        
        
        
        
        
    #rms()
    step2(lstm)   
    
    
    
    