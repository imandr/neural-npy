
"""
This is a batched LSTM forward and backward pass
"""
import numpy as np
from mynnet import Layer

class Recurrent(Layer):
  
    def __init__(self, inp, out_size,
                reversed = False, 
                last_row_only = False, weight_decay=1.e-5,
                name = None, applier = None):
        Layer.__init__(self, [inp], name)
        # input layer is expected to have output shape (sequence_length, in_size)
        assert len(inp.shape) == 2, "Input for Recurrent layer must be 2-dmensional (sequence, size)"
        self.NC = out_size
        self.WeightDecay = weight_decay
        self.LastRowOnly = last_row_only
        self.Reversed = reversed
        sequenceLength, self.Nin = inp.shape
        self.OutShape = (self.NC,) if self.LastRowOnly else (sequenceLength, self.NC) 
        self.State = None
        self.gState = None
    
    def initParams(self, rng = None):
        # in_shape is (sequence, nin)
        if rng == None: rng = np.random.RandomState()
        #print "Nin=", nin, "  NC=", self.NC

        """ 
        Initialize parameters of the LSTM (both weights and biases in one matrix) 
        One might way to have a positive fancy_forget_bias_init number (e.g. maybe even up to 5, in some papers)
        """
        # +1 for the biases, which will be the first row of WLSTM
        self.W = rng.normal(size=(self.Nin + self.NC + 1, self.NC),         # f then N
                scale= 1.0 / np.sqrt(self.Nin + self.NC))
        self.W[0,:] = 0 # initialize biases to zero
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
        
        U = np.zeros((n, b, xpcpb))        # input + S + bias
        U[:,:,0] = 1
        U[:,:,1:input_size+1] = X
        S=np.empty((n, b, d))
        Y = np.empty((n, b, self.NC))
        prevc = c0 if not c0 is None else np.zeros((b, d))
        C = np.empty((n, b, d))
        
        for t in xrange(n):
            S[t] = prevc
            U[t,:,-d:] = prevc
            C[t] = np.tanh(U[t].dot(self.W))
            prevc = C[t]
            
        self.Y = C.copy()
        cache = {}
        cache['C'] = C
        cache['S'] = S
        cache['c0'] = c0
        cache['U'] = U
        
        self.Cache = cache
        
        return self.Y, prevc
        
    def backward(self, dY, dcn = None, cache=None): 

        """
        dY_in should be of shape (n,b,output_size), where n = length of sequence, b = batch size
        """
        if self.Reversed:
            dY = dY[::-1,:,:]

        cache = self.Cache if cache is None else cache
        y = self.Y
        C = cache['C']
        S = cache['S']
        c0 = cache['c0']
        U = cache['U']
        
        n, b, _ = dY.shape
        d = self.NC # hidden size
        input_size = U.shape[2]-1-d        # - C - bias

        # views
        X = U[:,:,1:input_size+1]
        
        
        dW = np.zeros_like(self.W)
        
        dX = np.empty_like(X)
        dS = np.empty_like(S)
        
        Wx = self.W[1:1+input_size,:]
        Ws = self.W[1+input_size:,:]
        
        gTanh = 1.0 - C**2
        
        #gF = SmN*dFprime
        #gN = (1.0-Ff)*dNprime

        dC = dcn if not dcn is None else np.zeros((b, d))
        
        #if dcn is not None: dC[n-1] += dcn.copy() # carry over gradients from later
        for t in reversed(xrange(n)):
            gC = dC + dY[t]
            
            gZ = gC * gTanh[t]
            
            dXS = np.dot(gZ, self.W.T)       # [nb, nc] dot [nc, 1+nx+nc] -> [nb, 1+nx+nc]
            #print dXS.shape
            
            dX[t] = dXS[:,1:1+input_size]
            dS[t] = dXS[:,-d:]
            
            dW += np.dot(U[t].T, gZ)
                        
            dC = dS[t]
            
        return dX, dW, dC
    

#
# "Canonic" inerface: x and y are shaped as (batch, sequence, data)
#

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

    def fprop(self, x, state_in):
      x = x[0]
      #x = x.reshape((x.shape[0], -1, self.Nin))
      y, state = self.step(x, state_in)
      if self.LastRowOnly:
          y = y[:,-1,:]
      return y, state

    def bprop(self, x, state_in, y, state_out, gy, gState):
        # if LastRowOnly == True, gy must be (batch, NC)
        # otherwise (batch, sequence_length, NC)
        x = x[0]
        if self.LastRowOnly:
            l = x.shape[1]     # sequence length
            gy_expanded = np.zeros((gy.shape[0], l, gy.shape[1]))
            gy_expanded[:,-1,:] = gy
            gy = gy_expanded
        bsize = dy.shape[0]
        dcn = gState
        if dcn is None:
            dcn = np.zeros((bsize, self.NC))   # -> (batch, hidden)
        dx, dw, dcn = self.backward(dy.transpose((1,0,2)), dcn)
        dx = dx.transpose((1,0,2))
        
        return [dx], dcn, [dw]
  
    def getParams(self):
        return (self.W,)
        
    def setParams(self, tup):
        self.W, = tup

    def getParamsAdDict(self):
        return {    "W": self.W    }
        
    def setParamsAdDict(self, dct):
        self.W = dct["W"]

# -------------------
# TEST CASES
# -------------------

def create_cell(input_size, nbdo):
  from mynnet import InputLayer

  n,b,d = nbdo # sequence length, batch size, hidden size, output size
  
  inp = InputLayer((n,input_size))
  lstm = Recurrent(inp, d)
  lstm.initParams()
  return lstm
    
        
def checkSequentialMatchesBatch():

  input_size = 3
  nbd = (5, 17, 7)
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

