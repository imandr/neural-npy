
"""
This is a batched LSTM forward and backward pass
"""
import numpy as np
from mynnet import Layer

class LSTM_S(Layer):
  
    def __init__(self, inp, out_size, hidden_size, 
                reversed = False, 
                last_row_only = False, weight_decay=1.e-5,
                name = None, applier = None):
        Layer.__init__(self, [inp], name)
        # input layer is expected to have output shape (sequence_length, in_size)
        self.NC = hidden_size
        self.Nout = out_size
        self.SequenceLength = None
        self.SequenceLength, self.Nin = inp.shape
        self.WeightDecay = weight_decay
        self.LastRowOnly = last_row_only
        self.Reversed = reversed
        self.OutShape = (self.Nout,) if self.LastRowOnly else (self.SequenceLength, self.Nout) 
        #self.State = None
    
    def initParams(self, rng = None):
        # in_shape is (sequence, nin)
        if rng == None: rng = np.random.RandomState()
        #print "Nin=", nin, "  NC=", self.NC

        """ 
        Initialize parameters of the LSTM (both weights and biases in one matrix) 
        One might way to have a positive fancy_forget_bias_init number (e.g. maybe even up to 5, in some papers)
        """
        # +1 for the biases, which will be the first row of WLSTM
        self.W = rng.normal(size=(self.Nin + self.NC + 1, 2 * self.NC),         # f then N
                scale= 1.0 / np.sqrt(self.Nin + self.NC))
        self.V = rng.normal(size=(self.Nin + self.NC + 1, self.Nout),
                scale= 1.0 / np.sqrt(self.Nin + self.NC))
        self.W[0,:] = 0 # initialize biases to zero
        self.V[0,:] = 0 # initialize biases to zero
        # forget gates get little bit negative bias initially to encourage them to be turned off
        # remember that due to Xavier initialization above, the raw output activations from gates before
        # nonlinearity are zero mean and on order of standard deviation ~1
        #self.W[0,:self.NC] = 3.0
        
    def forward(self, X, state = None):
        c0 = state
        
        """
        X should be of shape (n,b,input_size), where n = length of sequence, b = batch size
        """

        if self.Reversed:
            X = X[::-1,:,:]

        n,b,input_size = X.shape
        d = self.NC # hidden size
        if c0 is None: c0 = np.zeros((b,d))
        
        xpcpb = input_size + 1 + d
        
        FN = np.zeros((n, b, d*2))          # forget and N
        FNf = np.zeros((n, b, d*2))         # after non-linearity
        U = np.zeros((n, b, xpcpb))        # input + C + bias
        U[:,:,0] = 1
        U[:,:,1:input_size+1] = X
        Z = np.empty((n, b, xpcpb))         # X and C
        Z[:,:,0] = 1    # for the bias
        Z[:,:,1:1+input_size] = X
        S=np.empty((n, b, d))
        Y = np.empty((n, b, self.Nout))
        prevc = c0 if not c0 is None else np.zeros((b, d))
        C = np.empty((n, b, d))
        for t in xrange(n):
            S[t] = prevc
            U[t,:,-d:] = prevc
            FN[t] = U[t].dot(self.W)
            FNf[t,:,:d] = 1.0/(1.0+np.exp(-FN[t,:,:d]))        # f: sigmoid
            FNf[t,:,d:] = np.tanh(FN[t,:,d:])                   # N: th
            
            Ft = FNf[t,:,:d]        # and this is view too
            Nt = FNf[t,:,d:]        # this is view
            
            C[t] = prevc*(1.0-Ft) + Nt*Ft
            prevc = C[t]
            
            Z[t,:,-d:] = C[t]
        y = np.dot(Z, self.V)
        self.Cache = dict(
            C=C, S=S, Z=Z, c0=c0, U=U, FN=FN, FNf=FNf, y=y
        )
        return y, prevc
        
    def backward(self, dY, dcn = None, cache=None): 

        """
        dY_in should be of shape (n,b,output_size), where n = length of sequence, b = batch size
        """
        if self.Reversed:
            dY = dY[::-1,:,:]

        cache = self.Cache if cache is None else cache
        y = cache['y']
        C = cache['C']
        S = cache['S']
        Z = cache['Z']
        c0 = cache['c0']
        U = cache['U']
        FN = cache['FN']
        FNf = cache['FNf']
        
        n, b, _ = dY.shape
        d = self.NC # hidden size
        input_size = U.shape[2]-1-d        # - C - bias

        # views
        X = U[:,:,1:input_size+1]
        Ff = FNf[:,:,:d]
        F = FN[:,:,:d]
        Nf = FNf[:,:,d:]
        N = FN[:,:,d:]
        
        #Wxf = self.W[1:1+input_size,:d]     # view
        #Wxn = self.W[1:1+input_size,d:]     # view
        #Wsf = self.W[1+input_size:,:d]     # view
        #Wsn = self.W[1+input_size:,d:]     # view
        
        Wx = self.W[1:1+input_size,:]
        Ws = self.W[1+input_size:,:]

        gW = np.zeros(self.W.shape)
        gV = np.zeros(self.V.shape)
        gX = np.empty_like(X)
        gS = np.empty_like(C)
        
        NmS = Nf - S
        
        #print "S:", S
        #print "N:", N

        dZ = np.dot(dY, self.V.T)           # [n, nb, nout] dot [nout, nin+1+nc] -> [n, nb, nin+1+nc]
        dC = dcn if not dcn is None else np.zeros((b, d))
        dFprime = Ff*(1.0-Ff)
        dNprime = 1.0-Nf**2

        gFN = np.empty((n, b, d*2))
        gFN[:,:,:d] = NmS*dFprime
        gFN[:,:,d:] = Ff*dNprime
        gF = gFN[:,:,:d]
        gN = gFN[:,:,d:]

        gCgFNt = np.empty((b, d*2))

        #if dcn is not None: dC[n-1] += dcn.copy() # carry over gradients from later
        for t in reversed(xrange(n)):
            gC = dC + dZ[t,:,-d:]

            gCgFNt[:,:d] = gC*gF[t]
            gCgFNt[:,d:] = gC*gN[t]
            
            gX[t] = np.dot(gCgFNt, Wx.T) + dZ[t,:,1:1+input_size]
            gS[t] = np.dot(gCgFNt, Ws.T) + gC*(1-Ff[t])

            gV += np.dot(Z[t].T, dY[t])        # [mb, in+1+nc] [mb, out]  -> 
            gW += np.dot(U[t].T, gCgFNt)
            
            dC = gS[t]
            
        return gX, gW, gV, gS[0]
    

    def regularize(self):
        self.W *= (1.0 - self.WeightDecay)
        self.V *= (1.0 - self.WeightDecay)
        
    def step(self, x, state=None):
        # X: (mb, length, inputs)
    
        x = x.transpose((1,0,2))
        y, state = self.forward(x, state)
        # y is returned as (length, mbatch, outputs)
        return y.transpose((1,0,2)), state
    
    def bprop(self, x, state_in, y, state_out, gy, gState):
        x = x[0]
        n_mb = len(x)
        # if LastRowOnly == True, gy must be (batch, Nout)
        # otherwise (batch, sequence_length, Nout)
        if self.LastRowOnly:
            l = x.shape[1]     # sequence length
            gy_expanded = np.zeros((gy.shape[0], l, gy.shape[1]))
            gy_expanded[:,-1,:] = gy
            gy = gy_expanded

        bsize = gy.shape[0]
        dcn = gState
        if dcn is None:
            dcn = np.zeros((bsize, self.NC))   # -> (batch, hidden)
        dx, dw, dv, dcn = self.backward(gy.transpose((1,0,2)), dcn)
        gx = dx.transpose((1,0,2))
        return [gx], dcn, [dw/n_mb, dv/n_mb]

    def fprop(self, x, state_in):
      x = x[0]
      x = x.reshape((x.shape[0], -1, self.Nin))
      y, state_out = self.step(x, state_in)
      if self.LastRowOnly:
          y = y[:,-1,:]
      return y, state_out
    
#
# ParamMixin
#      
    def getParams(self):
        return self.W, self.V
        
    def setParams(self, tup):
        self.W, self.V = tup
        
    def getParamsAsDict(self):
        return {"W":self.W, "V":self.V}
        
    def setParamsAsDict(self, dct):
        self.W, self.V = dct["W"], dct["V"]




# -------------------
# TEST CASES
# -------------------

def create_cell(input_size, nbdo):
    from mynnet import InputLayer
    n,b,d,o = nbdo
    inp = InputLayer((n,input_size))
    lstm = LSTM_S(inp, o, d)
    lstm.initParams()
    return lstm


def checkSequentialMatchesBatch():
  """ check LSTM I/O forward/backward interactions """


  n,b,d,o = (5, 3, 7, 2) # sequence length, batch size, hidden size, output size
  input_size = 10
  lstm = create_cell(input_size, (n,b,d,o))
  
  
  X = np.random.randn(n,b,input_size)
  
  c0 = np.random.randn(b,d)

  # sequential forward
  cprev = c0
  caches = [{} for t in xrange(n)]
  ys1 = np.zeros((n,b,o))
  for t in xrange(n):
    xt = X[t:t+1]
    y, cprev = lstm.forward(xt, cprev)
    caches[t] = lstm.Cache
    ys1[t] = y

  # sanity check: perform batch forward to check that we get the same thing
  ys2, _  = lstm.forward(X, c0)
  batch_cache = lstm.Cache
  assert np.allclose(ys1, ys2), 'Sequential and Batch forward don''t match!'

  # eval loss
  wrand = np.random.randn(*ys1.shape)
  loss = np.sum(ys1 * wrand)
  dy = wrand

  # get the batched version gradients
  BdX, BdW, BdV, Bdc0 = lstm.backward(dy, cache=batch_cache)

  # now perform sequential backward
  dX = np.zeros_like(X)
  dW = np.zeros_like(BdW)
  dV = np.zeros_like(BdV)
  dc0 = np.zeros_like(c0)
  dcnext = None
  for t in reversed(xrange(n)):
    dyt = dy[t].reshape(1, b, o)
    dx, dWt, dVt, dcprev = lstm.backward(dyt, dcnext, cache=caches[t])
    dcnext = dcprev

    dW += dWt # accumulate LSTM gradient
    dV += dVt
    dX[t] = dx[0]
    if t == 0:
      dc0 = dcprev

  # and make sure the gradients match
  print 'Making sure batched version agrees with sequential version: (should all be True)'
  #print "BdX:", BdX
  #print "dX:", dX
  print np.allclose(BdX, dX)
  print np.allclose(BdW, dW)
  print np.allclose(Bdc0, dc0)
  
def checkTrain():

  n,b,d,o = (5, 10, 400, 10) # sequence length, batch size, hidden size, output size
  input_size = 10
  
  lstm = create_cell(input_size, (n,b,d,o))
  
  X = np.random.randn(b,n,input_size)
  H = np.random.randn(b,n,o)
  h0, state = lstm.step(X)
  print H.shape, h0.shape
  print "loss before:", np.mean(np.square(h0-H))
  eta = 0.12
  for i in range(100):
    lstm.train(X, H, eta)
    eta *= 0.999
    h1, state = lstm.step(X)
    print "eta: %f,  loss after %d step: %.4f" %(eta, i+1, np.mean(np.square(h1-H)))
  


def checkBatchGradient():
  """ check that the batch gradient is correct """

  from mynnet import InputLayer

  n,b,d,o = (1, 4, 3, 7) # sequence length, batch size, hidden size, output size
  input_size = 10
  
  lstm = create_cell(input_size, (n,b,d,o))

  X = np.random.randn(n,b,input_size)
  c0 = np.random.randn(b,d)
  
  print "c0:", c0

  # batch forward backward
  H, Ct = lstm.forward(X, c0)
  wrand = np.random.randn(*H.shape)
  loss = np.sum(H * wrand) # weighted sum is a nice hash to use I think
  dH = wrand
  dX, dW, dV, dc0 = lstm.backward(dH)

  def fwd():
    h, _ = lstm.forward(X, c0)
    return np.sum(h * wrand)

  # now gradient check all
  delta = 1e-7
  rel_error_thr_warning = 1e-2
  rel_error_thr_error = 1
  tocheck = [X, lstm.W, lstm.V, c0]
  grads_analytic = [dX, dW, dV, dc0]
  names = ['X', 'W', 'V', 'c0']
  for j in xrange(len(tocheck)):
    mat = tocheck[j]
    dmat = grads_analytic[j]
    name = names[j]
    # gradcheck
    for i in xrange(mat.size):
      old_val = mat.flat[i]
      mat.flat[i] = old_val + delta
      loss0 = fwd()
      mat.flat[i] = old_val - delta
      loss1 = fwd()
      mat.flat[i] = old_val

      grad_analytic = dmat.flat[i]
      grad_numerical = (loss0 - loss1) / (2 * delta)

      if grad_numerical == 0 and grad_analytic == 0:
        rel_error = 0 # both are zero, OK.
        status = 'OK'
      elif abs(grad_numerical) < 1e-7 and abs(grad_analytic) < 1e-7:
        rel_error = 0 # not enough precision to check this
        status = 'VAL SMALL WARNING'
      else:
        rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
        status = 'OK'
        if rel_error > rel_error_thr_warning: status = 'WARNING'
        if rel_error > rel_error_thr_error: status = '!!!!! NOTOK'

      # print stats
      print '%s checking param %s index %s (val = %+8f), analytic = %+8f, numerical = %+8f, relative error = %+8f' \
            % (status, name, `np.unravel_index(i, mat.shape)`, old_val, grad_analytic, grad_numerical, rel_error)


def branch():
    
    nx = 2
    nc = 3
    ny = 4
    
    Vxy = np.random.random((nx, ny))
    Vsy = np.random.random((nc, ny))
    
    x0 = np.random.random((nx,))
    s0 = np.random.random((nc,))
    
    y_ = np.random.random((ny,))
    c_ = np.random.random((nc,))
    
    def f(x,s):
        return x.dot(Vxy)+s.dot(Vsy), s
        
    def loss(x,s,y_,c_):
        y, c = f(x,s)
        l = (np.sum(np.square(y-y_)) + np.sum(np.square(c-c_)))/2
        gc = c-c_
        gy = y-y_
        return l, gy, gc
        
    def grads(x, s, dy, dc):
        gx = dy.dot(Vxy.T)
        gs = dc + dy.dot(Vsy.T)
        
        gVx = np.outer(x, dy)
        gVs = np.outer(s, dy)
        
        return gx, gs, gVx, gVs
        
    y0, c0 = f(x0, s0)
    l0, gy0, gc0 = loss(x0, s0, y_, c_)
    gx0, gs0, gVx0, gVs0 = grads(x0, s0, gy0, gc0)
        
    delta = 1.e-5
    
    for px, gp, n in ((x0, gx0, "x"), (s0, gs0, "s"), (Vxy, gVx0, "Vx"), (Vsy, gVs0, "Vs")):
        grad = np.empty(px.shape)
        for i in range(px.size):
            saved = px.flat[i]
            px.flat[i] = saved - delta
            l1,_,_ = loss(x0, s0, y_, c_)
            px.flat[i] = saved + delta
            l2,_,_ = loss(x0, s0, y_, c_)
            px.flat[i] = saved
            
            grad.flat[i] = (l2-l1)/(2*delta)
        print n, grad, gp
        
            
            
            
    

if __name__ == "__main__":

  checkSequentialMatchesBatch()
  raw_input('check OK, press key to continue to gradient check')
  checkBatchGradient()
  #checkTrain()
  #print 'every line should start with OK. Have a nice day!'
  
  #branch()

