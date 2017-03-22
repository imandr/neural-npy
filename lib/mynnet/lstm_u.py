
"""
This is a batched LSTM forward and backward pass
"""
import numpy as np
from mynnet import Layer

class LSTM_U(Layer):
  
    def __init__(self, inp, out_size,
                reversed = False, 
                last_row_only = False, weight_decay=1.e-5,
                name = None, applier = None):
        Layer.__init__(self, [inp], name)
        # input layer is expected to have output shape (sequence_length, in_size)
        self.NC = out_size
        self.SequenceLength, self.Nin = inp.shape
        self.WeightDecay = weight_decay
        self.LastRowOnly = last_row_only
        self.Reversed = reversed
        self.OutShape = (self.NC,) if self.LastRowOnly else (self.SequenceLength, self.NC) 
        
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
        self.W = rng.normal(size=(self.Nin + self.NC + 1, 3 * self.NC),         # f, i, N
                scale= 1.0 / np.sqrt(self.Nin + self.NC))
        self.W[0,:] = 0 # initialize biases to zero
        # forget gates get little bit negative bias initially to encourage them to be turned off
        # remember that due to Xavier initialization above, the raw output activations from gates before
        # nonlinearity are zero mean and on order of standard deviation ~1
        #self.W[0,:self.NC] = 0.5
        
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
        
        FIN = np.zeros((n, b, d*3))          # f, i, N
        FINf = np.zeros((n, b, d*3))         # after non-linearity
        U = np.zeros((n, b, xpcpb))        # input + C + bias
        U[:,:,0] = 1
        U[:,:,1:input_size+1] = X
        S=np.empty((n, b, d))
        Y = np.empty((n, b, self.NC))
        prevc = c0 if not c0 is None else np.zeros((b, d))
        C = np.empty((n, b, d))
        for t in xrange(n):
            S[t] = prevc
            U[t,:,-d:] = prevc
            FIN[t] = U[t].dot(self.W)
            FINf[t,:,:-d] = 1.0/(1.0+np.exp(-FIN[t,:,:-d]))        # f: sigmoid
            FINf[t,:,-d:] = np.tanh(FIN[t,:,-d:])                   # N: th
            
            Ft = FINf[t,:,:d]        # and this is view too
            It = FINf[t,:,d:-d]        # and this is view too
            Nt = FINf[t,:,-d:]        # this is view
            
            C[t] = prevc*Ft + Nt*It
            prevc = C[t]
            
        y = C.copy()
        self.Cache = dict( C=C, S=S, c0=c0, U=U, FIN=FIN, FINf=FINf )
        return y, prevc
        
    def backward(self, dY, dcn = None, cache=None): 

        """
        dY_in should be of shape (n,b,output_size), where n = length of sequence, b = batch size
        """
        if self.Reversed:
            dY = dY[::-1,:,:]

        cache = self.Cache if cache is None else cache
        C = cache['C']
        S = cache['S']
        c0 = cache['c0']
        U = cache['U']
        FIN = cache['FIN']
        FINf = cache['FINf']
        
        n, b, _ = dY.shape
        d = self.NC # hidden size
        input_size = U.shape[2]-1-d        # - C - bias

        # views
        X = U[:,:,1:input_size+1]
        Ff = FINf[:,:,:d]
        F = FIN[:,:,:d]
        If = FINf[:,:,d:-d]
        I = FIN[:,:,d:-d]
        Nf = FINf[:,:,-d:]
        N = FIN[:,:,-d:]
        
        Wf = self.W[1:,:d]
        Wi = self.W[1:,d:-d]
        Wn = self.W[1:,-d:]

        Wx = self.W[1:1+input_size,:]
        Ws = self.W[1+input_size:,:]
        
        dW = np.zeros_like(self.W)
        
        dX = np.empty_like(X)
        dS = np.empty_like(S)
        
        dFprime = Ff*(1.0-Ff)
        dIprime = If*(1.0-If)
        dNprime = 1.0-Nf**2
        
        #gF = SmN*dFprime
        #gN = (1.0-Ff)*dNprime

        gFIN = np.empty((n, b, d*3))
        gFIN[:,:,:d] = S*dFprime
        gFIN[:,:,d:-d] = Nf*dIprime
        gFIN[:,:,-d:] = If*dNprime
        gF = gFIN[:,:,:d]
        gI = gFIN[:,:,d:-d]
        gN = gFIN[:,:,-d:]

        gCgFINt = np.empty((b, d*3))
        
        dC = dcn if not dcn is None else np.zeros((b, d))
        
        #if dcn is not None: dC[n-1] += dcn.copy() # carry over gradients from later
        for t in reversed(xrange(n)):
            gC = dC + dY[t]
            
            gCgFINt[:,:d] = gC * gF[t]
            gCgFINt[:,d:-d] = gC * gI[t]
            gCgFINt[:,-d:] = gC * gN[t]
            
            dX[t] = np.dot(gCgFINt, Wx.T)
            dS[t] = np.dot(gCgFINt, Ws.T) + gC*Ff[t]

            dW += np.dot(U[t].T, gCgFINt)
                        
            dC = dS[t]
            
        return dX, dW, dC
    

#
# "Canonic" inerface: x and y are shaped as (batch, sequence, data)
#

    def addDeltas(self, dW):
        self.W += dW
        
    def regularize(self):
        self.W *= (1.0 - self.WeightDecay)
        
    def step(self, x, state=None):
        # X: (mb, length, inputs)
    
        x = x.transpose((1,0,2))
        y, state = self.forward(x, state)
        # y is returned as (length, mbatch, outputs)
        return y.transpose((1,0,2)), state
    
#
# mynnet inetrface
#

    def bprop(self, x, state_in, y, state_out, gy, gState):
        x = x[0]
        n_mb = len(x)
        # if LastRowOnly == True, gy must be (batch, NC)
        # otherwise (batch, sequence_length, NC)
        if self.LastRowOnly:
            l = x.shape[1]     # sequence length
            gy_expanded = np.zeros((gy.shape[0], l, gy.shape[1]))
            gy_expanded[:,-1,:] = gy
            gy = gy_expanded
        dcn = gState
        bsize = gy.shape[0]
        if dcn is None:
            dcn = np.zeros((bsize, self.NC))   # -> (batch, hidden)
        dx, dw, dcn = self.backward(gy.transpose((1,0,2)), dcn)
        gx = dx.transpose((1,0,2))
        return [gx], dcn, [dw/n_mb]

    def fprop(self, x, state_in):
        x = x[0]
        x = x.reshape((x.shape[0], -1, self.Nin))
        y, state_out = self.step(x, state_in)
        if self.LastRowOnly:
          y = y[:,-1,:]
        return y, state_out

    def getParams(self):
        return (self.W,)
        
    def setParams(self, tup):
        self.W = tup[0]

    def getParamsAsDict(self):
        return {"W":self.W}
        
    def setParamsAsDict(self, dct):
        self.W = dct["W"]
      

# -------------------
# TEST CASES
# -------------------

def create_cell(input_size, nbdo):
  from mynnet import InputLayer

  n,b,d = nbdo # sequence length, batch size, hidden size, output size
  
  inp = InputLayer((n,input_size))
  
  lstm = LSTM_U(inp, d)
  lstm.initParams()
  return lstm
    
        
def checkSequentialMatchesBatch():
  """ check LSTM I/O forward/backward interactions """

  input_size = 3
  nbd = (5, 10, 7)
  n, b, d = nbd
  o = d
  lstm = create_cell(input_size, nbd)
  X = np.random.randn(n,b,input_size)
  
  c0 = np.random.randn(b,d)

  # sequential forward
  cprev = c0
  caches = [{} for t in xrange(n)]
  ys1 = np.zeros((n,b,d))
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
  BdX, BdW, Bdc0 = lstm.backward(dy, cache=batch_cache)

  # now perform sequential backward
  dX = np.zeros_like(X)
  dW = np.zeros_like(BdW)
  dc0 = np.zeros_like(c0)
  dcnext = None
  for t in reversed(xrange(n)):
    dyt = dy[t].reshape(1, b, o)
    dx, dWt, dcprev = lstm.backward(dyt, dcnext, cache=caches[t])
    dcnext = dcprev

    dW += dWt # accumulate LSTM gradient
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

  input_size = 10
  nbd = (5, 10, 7)
  n, b, d = nbd
  o = d
  lstm = create_cell(input_size, nbd)
  
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

  input_size = 3
  nbd = (5, 11, 7)
  n, b, d = nbd
  o = d
  lstm = create_cell(input_size, nbd)

  X = np.random.randn(n,b,input_size)
  c0 = np.random.randn(b,d)
  
  print "c0:", c0

  # batch forward backward
  H, Ct = lstm.forward(X, c0)
  wrand = np.random.randn(*H.shape)
  loss = np.sum(H * wrand) # weighted sum is a nice hash to use I think
  dH = wrand
  dX, dW, dc0 = lstm.backward(dH)

  def fwd():
    h, _ = lstm.forward(X, c0)
    return np.sum(h * wrand)

  # now gradient check all
  delta = 1e-7
  rel_error_thr_warning = 1e-2
  rel_error_thr_error = 1
  tocheck = [X, lstm.W, c0]
  grads_analytic = [dX, dW, dc0]
  names = ['X', 'W', 'c0']
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


  

if __name__ == "__main__":

  checkSequentialMatchesBatch()
  raw_input('check OK, press key to continue to gradient check')
  checkBatchGradient()
  #checkTrain()
  #print 'every line should start with OK. Have a nice day!'
  
  #branch()

