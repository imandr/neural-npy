
"""
This is a batched LSTM forward and backward pass
"""
import numpy as np
from mynnet import Layer, ParamMixin

class GRU(ParamMixin, Layer):
  
    def __init__(self, inp, out_size,
                reversed = False, 
                last_row_only = False, weight_decay=1.e-5,
                name = None, applier = None):
        Layer.__init__(self, inp, name)
        ParamMixin.__init__(self, applier)
        # input layer is expected to have output shape (sequence_length, in_size)
        self.NC = out_size
        self.Nin = self.InShape[-1]
        self.WeightDecay = weight_decay
        self.LastRowOnly = last_row_only
        self.Reversed = reversed
        #self.State = None
    
    def init_layer(self, rng = None):
        # in_shape is (sequence, nin)
        if rng == None: rng = np.random.RandomState()
        #print "Nin=", nin, "  NC=", self.NC

        """ 
        Initialize parameters of the LSTM (both weights and biases in one matrix) 
        One might way to have a positive fancy_forget_bias_init number (e.g. maybe even up to 5, in some papers)
        """
        # +1 for the biases, which will be the first row of WLSTM
        self.W = rng.normal(size=(self.Nin + 1 + self.NC, 3 * self.NC),         # f then N
                scale= 1.0 / np.sqrt(self.Nin + self.NC))
        self.W[0,:] = 0 # initialize biases to zero
        # forget gates get little bit negative bias initially to encourage them to be turned off
        # remember that due to Xavier initialization above, the raw output activations from gates before
        # nonlinearity are zero mean and on order of standard deviation ~1
        self.W[0,self.NC:-self.NC] = 3.0
        self.reset_grads()
        
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
        
        RF = np.zeros((n, b, d*3))          # forget and N
        RFf = np.zeros((n, b, d*3))         # after non-linearity
        
        N = np.empty((n,b,d))
        Nf = np.empty((n,b,d))
        
        U = np.empty((n, b, xpcpb))        # input + C + bias
        U[:,:,0] = 1
        U[:,:,1:input_size+1] = X
        
        V = np.empty((n, b, xpcpb))        # input + C + bias
        V[:,:,0] = 1
        V[:,:,1:input_size+1] = X
        
        Wrf = self.W[:,:-d]
        Wn = self.W[:,-d:]
        
        S=np.empty((n, b, d))
        Y = np.empty((n, b, self.NC))
        prevc = c0 if not c0 is None else np.zeros((b, d))
        C = np.empty((n, b, d))
        for t in xrange(n):
            S[t] = prevc
            U[t,:,-d:] = prevc
            RF[t] = U[t].dot(self.Wrf)
            RFf[t] = 1.0/(1.0+np.exp(-RF[t]))        # f: sigmoid
            
            V[t,:,-d:] = RFf[t,:d]*S[t]
            N[t] = V[t].dot(Wn)
            Nf[t] = np.tanh(N[t])
            C[t] = (1-RFf[t,d:])*S[t] + RFf[t,d:]*Nf[t]
            
            prevc = C[t]
            
        self.Y = C.copy()
        cache = {}
        cache['C'] = C
        cache['S'] = S
        cache['c0'] = c0
        cache['U'] = U
        cache['V'] = V
        cache['RF'] = RF
        cache['RFf'] = RFf
        cache['N'] = N
        cache['Nf'] = Nf
        
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
        V = cache['V']
        RF = cache['RF']
        RFf = cache['RFf']
        N = cache['N']
        Nf = cache['Nf']
        
        n, b, _ = dY.shape
        d = self.NC # hidden size
        input_size = U.shape[2]-1-d        # - C - bias

        # views
        X = U[:,:,1:input_size+1]
        Ff = FNf[:,:,:d]
        F = FN[:,:,:d]
        Nf = FNf[:,:,d:]
        N = FN[:,:,d:]
        
        Wf = self.W[1:,:d]
        Wn = self.W[1:,d:]

        Wx = self.W[1:1+input_size,:]
        Ws = self.W[1+input_size:,:]
        
        #Wxf = self.W[1:1+input_size,:d]     # view
        #Wxn = self.W[1:1+input_size,d:]     # view
        #Wsf = self.W[1+input_size:,:d]     # view
        #Wsn = self.W[1+input_size:,d:]     # view
        
        dW = np.zeros_like(self.W)
        
        dX = np.empty_like(X)
        dS = np.empty_like(S)
        
        NmS = Nf - S
        dFprime = Ff*(1.0-Ff)
        dNprime = 1.0-Nf**2
        
        #gF = SmN*dFprime
        #gN = (1.0-Ff)*dNprime

        gFN = np.empty((n, b, d*2))
        gFN[:,:,:d] = NmS*dFprime
        gFN[:,:,d:] = Ff*dNprime
        gF = gFN[:,:,:d]
        gN = gFN[:,:,d:]
        
        gCgFNt = np.empty((b, d*2))
        
        
        #print "S:", S
        #print "N:", N

        dC = dcn if not dcn is None else np.zeros((b, d))
        
        #if dcn is not None: dC[n-1] += dcn.copy() # carry over gradients from later
        for t in reversed(xrange(n)):
            gC = dC + dY[t]
            
            gCgFNt[:,:d] = gC*gF[t]
            gCgFNt[:,d:] = gC*gN[t]
                        
            dX[t] = np.dot(gCgFNt, Wx.T)
            dS[t] = np.dot(gCgFNt, Ws.T) + gC*(1-Ff[t])

            dW += np.dot(U[t].T, gCgFNt)
                        
            dC = dS[t]
            
        return dX, dW, dC
    

#
# "Canonic" inerface: x and y are shaped as (batch, sequence, data)
#

    def reset_grads(self):
        self.gW = None
        self.gX = None
        self.gY = None

    def backprop(self, dy, dcn):
        bsize = dy.shape[0]
        if dcn is None:
          dcn = np.zeros((bsize, self.NC))   # -> (batch, hidden)
        dx, dw, dcn = self.backward(dy.transpose((1,0,2)), dcn)

        if self.gW is None:
            self.gW = dw
        else:
            self.gW += dw
            

        #print "gw, gv, gb, eta:", np.mean(np.square(self.gWLSTM)), np.mean(np.square(self.gV)), np.mean(np.square(self.gB))
        #print "dw, dv, db:", np.mean(np.square(self.dWLSTM)), np.mean(np.square(self.dV)), np.mean(np.square(self.dB))

        return dx.transpose((1,0,2)), dcn

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

    def bprop(self, gy):
        # if LastRowOnly == True, gy must be (batch, NC)
        # otherwise (batch, sequence_length, NC)
        if self.LastRowOnly:
            l = self.X.shape[1]     # sequence length
            gy_expanded = np.zeros((gy.shape[0], l, gy.shape[1]))
            gy_expanded[:,-1,:] = gy
            gy = gy_expanded
        gx, self.dState = self.backprop(gy, self.dState)
        if self.gY is None:
            self.gY = gy
            self.gX = gx
        else:
            self.gY += gy
            self.gX += gx
        return gx
  
    def fprop(self, x):
      x = x.reshape((x.shape[0], -1, self.Nin))
      y, self.State = self.step(x, self.State)
      if self.LastRowOnly:
          y = y[:,-1,:]
      return y

    def output_shape(self, input_shape):
        return (self.NC,) if self.LastRowOnly else (self.SequenceLength, self.NC) 
        
    def reset_state(self):
        #print "%s: reset state" % (self,)
        self.State = None
        self.dState = None
    
#
# ParamMixin
#      
    def __get_params(self):
        return (self.W,)
        
    def __set_params(self, tup):
        self.W = tup[0]

    params = property(__get_params, __set_params)
    
    def param_incs(self):
        return (self.dW,)

    def param_grads(self):
        return (self.gW,)

    def dump_params(self, prefix):
        return {
            prefix+"/W":    self.W,
        }

    def restore_params(self, prefix, dct):
        self.W = dct[prefix+"/W"]
      

def lstm_s(inp, out_size, hidden_size, 
                name = None, 
                reversed = False, 
                last_row_only = False, weight_decay=1.e-5,
                applier = None):
    from mynnet import Concat, Linear, LastRow
    r = LSTM_T(inp, hidden_size,
        name = None if name is None else name + "/lstm_t",
        reversed = reversed, weight_decay=weight_decay,
        applier = applier
        )
    c = Concat((inp, r), axis=1,
        name = None if name is None else name + "/concat",
        )
    x = c
    if last_row_only:
        x = LastRow(c,
            name = None if name is None else name + "/last_row",
            )
    l = Linear(x, out_size,
        name = None if name is None else name + "/linear",
        )
    return l

# -------------------
# TEST CASES
# -------------------

def create_cell(input_size, nbdo):
  from mynnet import InputLayer

  n,b,d = nbdo # sequence length, batch size, hidden size, output size
  
  inp = InputLayer((n,input_size))
  
  lstm = LSTM_S(inp, d, d+5)
  lstm.init_layer()
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

