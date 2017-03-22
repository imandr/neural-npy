"""
This is a batched LSTM forward and backward pass
"""
import numpy as np
from mynnet import Layer, InputLayer

class LSTM(Layer):
  
    def __init__(self, inp, out_size,
                reversed = False, 
                last_row_only = False, weight_decay=1.e-5,
                name = None, applier = None):
        Layer.__init__(self, [inp], name)
        # input layer is expected to have output shape (sequence_length, in_size)
        hidden_size = out_size
        self.NC = hidden_size
        self.Nout = out_size
        self.SequenceLength = None
        self.SequenceLength, self.Nin = inp.shape
        self.WeightDecay = weight_decay
        self.LastRowOnly = last_row_only
        self.Reversed = reversed
        self.OutShape = (self.Nout,) if self.LastRowOnly else (self.SequenceLength, self.Nout) 
        
    
    def initParams(self, rng = None):
        # in_shape is (sequence, nin)
        if rng == None: rng = np.random.RandomState()
        #print "Nin=", nin, "  NC=", self.NC

        """ 
        Initialize parameters of the LSTM (both weights and biases in one matrix) 
        One might way to have a positive fancy_forget_bias_init number (e.g. maybe even up to 5, in some papers)
        """
        # +1 for the biases, which will be the first row of WLSTM
        self.WLSTM = rng.normal(size=(self.Nin + self.NC + 1, 4 * self.NC),
                scale= 1.0 / np.sqrt(self.Nin + self.NC))
        #print "WLSTM shape=", self.WLSTM.shape
        self.WLSTM[0,:] = 0 # initialize biases to zero
        self.W = self.WLSTM
        # forget gates get little bit negative bias initially to encourage them to be turned off
        # remember that due to Xavier initialization above, the raw output activations from gates before
        # nonlinearity are zero mean and on order of standard deviation ~1
        self.WLSTM[0,self.NC:2*self.NC] = 3.0
        
    def forward(self, X, state = None):
        c0, h0 = state if not state is None else (None, None)
        
        """
        X should be of shape (n,b,input_size), where n = length of sequence, b = batch size
        """

        if self.Reversed:
            X = X[::-1,:,:]

        n,b,input_size = X.shape
        d = self.NC # hidden size
        if c0 is None: c0 = np.zeros((b,d))
        if h0 is None: h0 = np.zeros((b,d))
        
        # Perform the LSTM forward pass with X as the input
        xphpb = self.WLSTM.shape[0] # x plus h plus bias, lol
        Hout = np.zeros((n, b, d)) # hidden representation of the LSTM (gated cell content)
        #print "Hout size:", Hout.shape
        IFOG = np.zeros((n, b, d * 4)) # input, forget, output, gate (IFOG)
        IFOGf = np.zeros((n, b, d * 4)) # after nonlinearity
        C = np.zeros((n, b, d)) # cell content
        Ct = np.zeros((n, b, d)) # tanh of cell content
        Hin = np.empty((n, b, xphpb)) # input [1, xt, ht-1] to each tick of the LSTM
        Hin[:,:,1:input_size+1] = X
        Hin[:,:,0] = 1 # bias
        prevh = h0
        for t in xrange(n):
            # concat [x,h] as input to the LSTM
            #print "Hin shape:", Hin.shape, "   X shape=",X[t].shape, "   input_size=", input_size
            #Hin[t,:,1:input_size+1] = X[t]
            Hin[t,:,input_size+1:] = prevh
            # compute all gate activations. dots: (most work is this line)
            IFOG[t] = Hin[t].dot(self.WLSTM)
            # non-linearities
            IFOGf[t,:,:3*d] = 1.0/(1.0+np.exp(-IFOG[t,:,:3*d])) # sigmoids; these are the gates
            IFOGf[t,:,3*d:] = np.tanh(IFOG[t,:,3*d:]) # tanh
            # compute the cell activation
            prevc = C[t-1] if t > 0 else c0
            C[t] = IFOGf[t,:,:d] * IFOGf[t,:,3*d:] + IFOGf[t,:,d:2*d] * prevc
            Ct[t] = np.tanh(C[t])
            Hout[t] = IFOGf[t,:,2*d:3*d] * Ct[t]
            prevh = Hout[t]
        
        cache = {}
        cache['Hout'] = Hout
        cache['IFOGf'] = IFOGf
        cache['IFOG'] = IFOG
        cache['C'] = C
        cache['Ct'] = Ct
        cache['Hin'] = Hin
        cache['c0'] = c0
        
        # return C[t], as well so we can continue LSTM with prev state init if needed
        y = Hout
        cache['y'] = y
        self.Cache = cache
        
        return y, (C[t], Hout[t])
        
    def backward(self, dY_in, dcn = None, dhn = None, cache=None): 

        """
        dY_in should be of shape (n,b,output_size), where n = length of sequence, b = batch size
        """
        if self.Reversed:
            dY_in = dY_in[::-1,:,:]

        cache = self.Cache if cache is None else cache
        y = cache['y']
        Hout = cache['Hout']
        IFOGf = cache['IFOGf']
        IFOG = cache['IFOG']
        C = cache['C']
        Ct = cache['Ct']
        Hin = cache['Hin']
        c0 = cache['c0']

        dHout = dY_in.copy()

        n,b,d = Hout.shape
        input_size = self.WLSTM.shape[0] - d - 1 # -1 due to bias

        # backprop the LSTM
        dIFOG = np.zeros(IFOG.shape)
        dIFOGf = np.zeros(IFOGf.shape)
        dWLSTM = np.zeros(self.WLSTM.shape)
        dHin = np.zeros(Hin.shape)
        dC = np.zeros(C.shape)
        dX = np.zeros((n,b,input_size))
        dh0 = np.zeros((b, d))
        dc0 = np.zeros((b, d))
        # IM: already a copy dHout = dHout_in.copy() # make a copy so we don't have any funny side effects
        if dcn is not None: dC[n-1] += dcn.copy() # carry over gradients from later
        
        #print "backward:",dhn.shape, dHout[n-1].shape
        
        if dhn is not None: dHout[n-1] += dhn.copy()
        for t in reversed(xrange(n)):
        
            tanhCt = Ct[t]
            dIFOGf[t,:,2*d:3*d] = tanhCt * dHout[t]     # [nb, Nc] * [nb, Nc] -> [nb, Nc]
            # backprop tanh non-linearity first then continue backprop
            dC[t] += (1-tanhCt**2) * (IFOGf[t,:,2*d:3*d] * dHout[t])
        
            if t > 0:
                dIFOGf[t,:,d:2*d] = C[t-1] * dC[t]
                dC[t-1] += IFOGf[t,:,d:2*d] * dC[t]
            else:
                dIFOGf[t,:,d:2*d] = c0 * dC[t]
                dc0 = IFOGf[t,:,d:2*d] * dC[t]
            dIFOGf[t,:,:d] = IFOGf[t,:,3*d:] * dC[t]
            dIFOGf[t,:,3*d:] = IFOGf[t,:,:d] * dC[t]
        
            # backprop activation functions
            dIFOG[t,:,3*d:] = (1 - IFOGf[t,:,3*d:] ** 2) * dIFOGf[t,:,3*d:]
            y = IFOGf[t,:,:3*d]
            dIFOG[t,:,:3*d] = (y*(1.0-y)) * dIFOGf[t,:,:3*d]
        
            # backprop matrix multiply
            dWLSTM += np.dot(Hin[t].transpose(), dIFOG[t])
            dHin[t] = dIFOG[t].dot(self.WLSTM.T)
        
            # backprop the identity transforms into Hin
            dX[t] = dHin[t,:,1:input_size+1]
            if t > 0:
                dHout[t-1,:] += dHin[t,:,input_size+1:]
            else:
                dh0 += dHin[t,:,input_size+1:]

        return dX, dWLSTM, dc0, dh0
    

#
# mynnet inetrface
#

    def step(self, x, state=None):
        # X: (mb, length, inputs)
    
        x = x.transpose((1,0,2))
        y, state = self.forward(x, state)
        # y is returned as (length, mbatch, outputs)
        return y.transpose((1,0,2)), state
    
    def fprop(self, x, in_state):
        x = x[0]
        #print "LSTM.fprop: x=", x.shape
        x = x.reshape((x.shape[0], -1, self.Nin))
        y, state = self.step(x, in_state)
        if self.LastRowOnly:
            y = y[:,-1,:]
        return y, state

    def bprop(self, x, state_in, y, state_out, gy, gState):
        x = x[0]
        mb = x.shape[0]
        # if LastRowOnly == True, gy must be (batch, Nout)
        # otherwise (batch, sequence_length, Nout)
        if self.LastRowOnly:
            l = x.shape[1]     # sequence length
            gy_expanded = np.zeros((gy.shape[0], l, gy.shape[1]))
            gy_expanded[:,-1,:] = gy
            gy = gy_expanded
            
        bsize = gy.shape[0]
        if gState is None:
            dcn = np.zeros((bsize, self.NC))   # -> (batch, hidden)
            dhn = np.zeros((bsize, self.NC))
        else:
            dcn, dhn = gState    
        
        dx, dw, dcn, dhn = self.backward(gy.transpose((1,0,2)), dcn, dhn)
        gx = dx.transpose((1,0,2))
        return [gx], (dcn, dhn), [dw/mb]
#
# "Canonic" inerface: x and y are shaped as (batch, sequence, data)
#

    def regularize(self):
        self.WLSTM *= (1.0 - self.WeightDecay)

    def getParams(self):
        return (self.WLSTM,)
        
    def setParams(self, tup):
        self.WLSTM, = tup

    def getParamsAsDict(self):
        return {"WLSTM":self.WLSTM}
        
    def setParamsAsDict(self, dct):
        self.WLSTM = dct["WLSTM"]
        
def create_cell(input_size, nbdo):
  from mynnet import InputLayer

  n,b,d,o = nbdo # sequence length, batch size, hidden size, output size
  
  inp = InputLayer((n,input_size))
  lstm = LSTM(inp, d)
  lstm.initParams()
  return lstm
      
def checkSequentialMatchesBatch():
  """ check LSTM I/O forward/backward interactions """

  n,b,d,o = (5, 3, 7, 2) # sequence length, batch size, hidden size, output size
  input_size = 10
  
  lstm = create_cell(input_size, (n,b,d,o))
  X = np.random.randn(n,b,input_size)
  h0 = np.random.randn(b,d)
  c0 = np.random.randn(b,d)

  # sequential forward
  cprev = c0
  hprev = h0
  caches = [{} for t in xrange(n)]
  Hcat = np.zeros((n,b,d))
  for t in xrange(n):
    xt = X[t:t+1]
    y, (cprev, hprev) = lstm.forward(xt, (cprev, hprev))
    caches[t] = lstm.Cache
    Hcat[t] = y

  # sanity check: perform batch forward to check that we get the same thing
  H, _  = lstm.forward(X, (c0, h0))
  batch_cache = lstm.Cache
  assert np.allclose(H, Hcat), 'Sequential and Batch forward don''t match!'

  # eval loss
  wrand = np.random.randn(*Hcat.shape)
  loss = np.sum(Hcat * wrand)
  dH = wrand

  # get the batched version gradients
  BdX, BdWLSTM, Bdc0, Bdh0 = lstm.backward(dH, cache=batch_cache)

  # now perform sequential backward
  dX = np.zeros_like(X)
  dWLSTM = np.zeros_like(BdWLSTM)
  dc0 = np.zeros_like(c0)
  dh0 = np.zeros_like(h0)
  dcnext = None
  dhnext = None
  for t in reversed(xrange(n)):
    dht = dH[t].reshape(1, b, d)
    dx, dWLSTMt, dcprev, dhprev = lstm.backward(dht, dcnext, dhnext, cache=caches[t])
    dhnext = dhprev
    dcnext = dcprev

    dWLSTM += dWLSTMt # accumulate LSTM gradient
    dX[t] = dx[0]
    if t == 0:
      dc0 = dcprev
      dh0 = dhprev

  # and make sure the gradients match
  print 'Making sure batched version agrees with sequential version: (should all be True)'
  print np.allclose(BdX, dX)
  print np.allclose(BdWLSTM, dWLSTM)
  print np.allclose(Bdc0, dc0)
  print np.allclose(Bdh0, dh0)
  
def checkTrain():

  n,b,d,o = (5, 3, 7, 2) # sequence length, batch size, hidden size, output size
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

  # lets gradient check this beast
  n,b,d,o = (5, 3, 7, 2) # sequence length, batch size, hidden size, output size
  input_size = 10
  
  lstm = create_cell(input_size, (n,b,d,o))

  X = np.random.randn(n,b,input_size)
  h0 = np.random.randn(b,d)
  c0 = np.random.randn(b,d)

  # batch forward backward
  H, (Ct, Ht) = lstm.forward(X, (c0, h0))
  wrand = np.random.randn(*H.shape)
  loss = np.sum(H * wrand) # weighted sum is a nice hash to use I think
  dH = wrand
  dX, dWLSTM, dc0, dh0 = lstm.backward(dH)

  def fwd():
    h,_ = lstm.forward(X, (c0, h0))
    return np.sum(h * wrand)

  # now gradient check all
  delta = 1e-5
  rel_error_thr_warning = 1e-2
  rel_error_thr_error = 1
  tocheck = [X, lstm.WLSTM, c0, h0]
  grads_analytic = [dX, dWLSTM, dc0, dh0]
  names = ['X', 'WLSTM', 'c0', 'h0']
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
  #raw_input('check OK, press key to continue to gradient check')
  checkBatchGradient()
  #checkTrain()
  print 'every line should start with OK. Have a nice day!'