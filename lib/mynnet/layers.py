import numpy as np
import math, time, random

from convnet.conv import convolve_xw, convolve_xy
from convnet.pool import pool, bprop_pool

class Layer(object):
    
    def __init__(self, inputs, name=None):
        self.Name = name
        self.Inputs = inputs[:]
        self.StepId = None
        self.State = None
        self.NFeedBacks = 0
        self.NSuccessors = 0
        for inp in inputs:
            inp.addSuccessor(self)
        self.Applier = None
        self.OutShape = []
        self.gParams = None
        self.gX = None
        self.gState = None
        self.X = None
        self.Y = None
        self.State = None
        self.BackPropApplied = None
            
    def __str__(self):
        return "<Layer %s>" % (self.Name or self.__class__)
        
    #
    # Basic API
    #
    
    def fprop(self, xes, state_in):
        """ Calculate layer output for given input (forward propagation). 
            returns (Y, new_state)
        """
        raise NotImplementedError()

    def bprop(self, xes, state_in, y, state_out, gY, gState):
        """ Calculate gradients.
             returns ([gX], gState_in, [gParams/mbatch])
        """
        raise NotImplementedError()
        
    def setParams(self, params):      # override this
        pass
        
    def getParams(self):      # override this
        return []
        
    def _setParams(self, params):
        return self.setParams(params)
        
    def _getParams(self):
        return self.getParams()
        
    params = property(_getParams, _setParams)
    
    def getParamsAsDict(self):      # override this
        return {}
        
    def setParamsAsDict(self, dct):      # override this
        pass
        
    def _getParamsAsDict(self):
        return self.getParamsAsDict()
        
    def _setParamsAsDict(self, dct):
        self.setParamsAsDict(dct)
        
    paramsAsDict = property(_getParamsAsDict, _setParamsAsDict)

    @property
    def shape(self):
        # returns output shape, without minibatch component, as a tuple
        return self.OutShape
        
    def initParams(self, rng=None):      # override this
        pass
        
    def resetState(self):      # override this
        self.State = None
        self.gState = None
        
    def updateParams(self, deltas):
        for ip, p in enumerate(self.params):
            #print "%s: updating param %d with delta: %s" % (self, ip, deltas[ip])
            p += deltas[ip]
        
    def regularize(self):      # override this
        pass
        
    #
    # Higher level API
    #
    
    def init(self, rng = None):
        self.initParams(rng)
        if self.Inputs:
            for i in self.Inputs:
                i.init(rng)

    def addSuccessor(self, layer):
        self.NSuccessors += 1
        
    def reset_state_rec(self):
        self.resetState()
        for inp in self.Inputs:
            inp.reset_state_rec()

    def compute(self, step_id = None):
        if step_id is None:
            step_id = time.time()
        if self.StepId != step_id:
            self.X = [l.compute(step_id) for l in self.Inputs]
            self.StateIn = self.State
            y, self.State = self.fprop(self.X, self.State)
            self.Y = y
            self.StepId = step_id
            #self.resetBackprop()
            #print "compute(%s): inputs:%s -> y:%s" % (self, inputs, y)
        #print "%s compute(): Y=%s" % (self, self.Y)
        return self.Y

    def resetBackprop(self):
        #print "%s:resetBackprop()" % (self,)
        self.NFeedBacks = 0
        if self.gParams is None:
            self.gParams = [np.zeros_like(p) for p in self.params]   
        else:
            for gp in self.gParams:
                gp[...] = 0.0
        self.gX = None
        self.gY = None
        
    def reset_backprop_rec(self):
        self.resetBackprop()
        for inp in self.Inputs:
            inp.reset_backprop_rec()

    def backprop(self, gY, step_id, normalize_grads = False):
        assert self.StepId == step_id, "%s: step id mismatch" % (self,)
        #print "backprop: %s" % (self,)
        self.BackPropApplied = None
        
        if self.gY is None:
            self.gY = gY.copy()
        else:
            self.gY += gY
        
        self.NFeedBacks += 1
        assert self.NFeedBacks <= self.NSuccessors
        if self.NFeedBacks < self.NSuccessors:
            return
            
        self.gX, self.gState, self.gParams = self.bprop(self.X, self.StateIn, self.Y, self.State, gY, self.gState)
        gx_norm = 1.0
        if normalize_grads:
            gy_rms = np.sum(np.square(self.gY))
            gx_rms = np.sum(np.square(self.gX))
            gx_norm = (1.0+math.sqrt(gy_rms/gx_rms))/2
            #gx_norm = math.sqrt(math.sqrt(gy_rms/gx_rms))
            #print "%s: gy rms: %f, gx rms: %f, norm: %f" % (self, gy_rms, gx_rms, gx_norm)
        for ix, inp in enumerate(self.Inputs):
            inp.backprop(self.gX[ix]*gx_norm, step_id, normalize_grads=normalize_grads)
        #print "%s.backprop: gX = %s" % (self, self.gX)
                
    def param_grads(self):
        return self.gParams
                
    def applyBackprop(self, eta, step_id):
        # insert applier here
        
        if self.params and eta and step_id != self.BackPropApplied:

            assert self.NFeedBacks == self.NSuccessors, "%s: number of collected feedbacks (%d) does not match number of successors (%d)" % (self, self.NFeedBacks, self.NSuccessors)
            deltas = self.gradsToDeltas(eta, self.gParams)
            self.updateParams(deltas)
            #print "%s: regularize()" % (self,)
            self.regularize()
            self.resetBackprop()
            self.BackPropApplied = step_id

    def apply_backprop_rec(self, eta, step_id):
        self.applyBackprop(eta, step_id)
        for inp in self.Inputs:
            inp.apply_backprop_rec(eta, step_id)

    def setApplier(self, a):
        self.Applier = a

    def gradsToDeltas(self, eta, grads):
        
        if eta == 0.0:  # no updates
            return tuple([np.zeros(g.shape) for g in grads])

        applier = self.Applier
        if applier:
            deltas = applier(eta, grads)
        else:
            deltas = [-eta*g for g in grads]
        return deltas

    def pruneStats(self):
        return []

        
    def checkGradients(self, mb=3, limit=10):
        x = [np.random.random((mb,)+inp.shape)-0.5 for inp in self.Inputs]
        y, out_state = self.fprop(x, None)
        in_state = None
        if not out_state is None:
            if isinstance(out_state, (list, tuple)):
                in_state = [np.random.random(s.shape)-0.5 for s in out_state]
                if isinstance(out_state, tuple):
                    in_state = tuple(in_state)
            else:
                in_state = np.random.random(out_state.shape)
        if not in_state is None:
            y, out_state = self.fprop(x, in_state)
        y_ = np.random.random(y.shape)-0.5
        
        def loss(y_, x, in_state):
            #print y.shape[0]
            y, _ = self.fprop(x, in_state)
            return np.sum(np.square(y-y_))/2/y.shape[0], (y - y_)/y.shape[0]
        
        loss0, gy = loss(y_, x, in_state)
        gx, gstate, gparams = self.bprop(x, in_state, y, out_state, gy, None)
        
        delta = 0.001
        # check x
        print "--- dL/dx ---"
        for ix, xx in enumerate(x):
            j_lst = list(range(xx.size))
            if not limit is None and limit < len(j_lst):
                j_lst = sorted(random.sample(j_lst, limit))
            
            for j in j_lst:
                xsave = xx.flat[j]
                xx.flat[j] = xsave - delta
                l0, _ = loss(y_, x, in_state)
                xx.flat[j] = xsave + delta
                l1, _ = loss(y_, x, in_state)
                xx.flat[j] = xsave
                dldxi = (l1-l0)/(2*delta)
                ok = abs(gx[ix].flat[j]/dldxi - 1.0) < 0.01
                j_unravel = np.unravel_index(j, xx.shape)
                print "dl/dx(%d,%s): analitic: %f, calculated: %f, analitic/calculated: %f %s" % \
                    (ix, j_unravel, gx[ix].flat[j], dldxi, gx[ix].flat[j]/dldxi,
                    "<--- error" if not ok else "")

        # checl params
        print "--- dL/dp ---"
        for ip, p in enumerate(self.params):
            pflat = p.flat
            j_lst = list(range(p.size))
            if not limit is None and limit < len(j_lst):
                j_lst = sorted(random.sample(j_lst, limit))
            for j in j_lst:
                psave = pflat[j]
                pflat[j] = psave - delta
                l0, _ = loss(y_, x, in_state)
                pflat[j] = psave + delta
                l1, _ = loss(y_, x, in_state)
                pflat[j] = psave
                dldpi = (l1-l0)/(2*delta)/mb
                ok = abs(gparams[ip].flat[j]/dldpi - 1.0) < 0.01
                j_unravel = np.unravel_index(j, p.shape)
                print "dl/dp(%d,%s): analitic: %f, calculated: %f, analitic/calculated: %f %s" % \
                    (ip, j_unravel, gparams[ip].flat[j], dldpi, gparams[ip].flat[j]/dldpi,
                    "<--- error" if not ok else "")
                
        # check state
        if not gstate is None:
            print "--- dL/ds ---"
            in_state_lst = in_state if isinstance(in_state, (list, tuple)) else [in_state]
            gstate_lst = gstate if isinstance(gstate, (list, tuple)) else [gstate]
            for iq, q in enumerate(in_state_lst):
                qflat = q.flat
                j_lst = list(range(q.size))
                #print iq, q.size
                if not limit is None and limit < len(j_lst):
                    j_lst = sorted(random.sample(j_lst, limit))
                for j in j_lst:
                    qsave = qflat[j]
                    qflat[j] = qsave - delta
                    l0, _ = loss(y_, x, in_state)
                    qflat[j] = qsave + delta
                    l1, _ = loss(y_, x, in_state)
                    qflat[j] = qsave
                    dldqi = (l1-l0)/(2*delta)
                    ok = abs(gstate_lst[iq].flat[j]/dldqi - 1.0) < 0.01
                    print "dl/dstate(%d,%d): analitic: %f, calculated: %f, analitic/calculated: %f %s" % \
                        (iq, j, gstate_lst[iq].flat[j], dldqi, gstate_lst[iq].flat[j]/dldqi,
                        "<--- error" if not ok else "")
            
                

class Concat(Layer):
    
    # concatenates along specified axis (mb, axis0, axis1, axis2,...)
    # assumes all other dimensions except "axis" are the same
    # remember that input.output_shape() returns shape without the minibatch dimension
    
    def __init__(self, inputs, axis=0, name=None):
        Layer.__init__(self, inputs, name)
        self.Axis = axis            # axis without including the minibatch dimension
        self.InShapes = [i.shape for i in self.Inputs]
        s0 = self.InShapes[0]
        self.Splits = []
        j = 0
        dim = 0
        for i, s in enumerate(self.InShapes):
            assert s[:axis] == s0[:axis] and s[axis+1:] == s0[axis+1:], \
                    "input %s does not have same shape as input 0: %s vs. %s" % (i, s, s0)
            if i > 0:
                self.Splits.append(j)
            j += s[axis]
            dim += s[axis]
        out_shape = list(s0)
        out_shape[axis] = dim
        self.OutShape = tuple(out_shape)
        
    def fprop(self, xes, state_in):
        return np.concatenate(xes, axis=self.Axis+1), None      # +1 for the minibatch dim

    def bprop(self, xes, state_in, y, state_out, gY, gState):
        """ Calculate gradients.
             returns ([gX], gState_in, [gParams])
        """
        gy_split = np.split(gY, self.Splits, axis=self.Axis+1)      # +1 for the minibatch
        return gy_split, None, []
        
class Reshape(Layer):
    
    def __init__(self, inp, out_shape, name=None):
        Layer.__init__(self, [inp], name)
        assert np.prod(inp.shape) == np.prod(out_shape)
        self.OutShape = out_shape

    def fprop(self, x, in_state):
        return x[0].reshape((-1,)+self.OutShape), None
        
    def bprop(self, x, state_in, y, state_out, gY, gState):
        x = x[0]
        return [gY.reshape(x.shape)], None, []
        
class LastRow(Layer):
    
    def __init__(self, inp, name=None):
        Layer.__init__(self, [inp], name)
        self.OutShape = (inp.shape[-1],)

    def fprop(self, x, in_state):
        x = x[0]
        #print "LastRow.fprop: input x:", x
        x = x.reshape((x.shape[0], -1, x.shape[-1]))
        #print "LastRow.fprop: reshaped x:", x
        y = x[:, -1, :]
        #print "LastRow.fprop: y:", y
        
        return y, None
        
    def bprop(self, x, state_in, y, state_out, gY, gState):
        x = x[0]
        gx = np.zeros_like(x).reshape((x.shape[0], -1, x.shape[-1]))
        gx[:,-1,:] = gY
        return [gx.reshape(x.shape)], None, []
        
class InputLayer(Layer):
    
    def __init__(self, shape, name=None):
        Layer.__init__(self, [], name)
        # shape without minibatch dim
        #print "shape=", shape
        self.InShape = shape
        self.OutShape = shape
        self.StepId = None
        self.StateIn = None
        self.State = None
    
    def set(self, x, step_id):
        for i, d in enumerate(x.shape[1:]):
            myd = self.InShape[i]
            if not myd is None:
                assert d == myd, "Invalid input for input layer %s: %s, expected %s" % (
                    self, x.shape[1:], self.InShape)
        #print "InputLayer(%s).set(%s)" % (self, x)
        self.X = x
        self.StepId = step_id
        #print "%s.set(): self.X=%s" % (self, self.X)
        
        
    def compute(self, step_id = None):
        assert step_id == self.StepId, "Step id mismatch: %s vs. %s" % (self.StepId, step_id)
        self.Y = self.X
        return self.Y

    def fprop(self, x, state_in):
        #print "%s.fprop: returning %s" % (self, self.X)
        return self.X, None
        
    def bprop(self, x, state_in, y, state_out, gY, gState):
        return [gY], None, []
        
class Transpose(Layer):
    
    def __init__(self, inp, t, name=None):
        Layer.__init__(self, [inp], name)
        self.InShape = inp.shape
        self.T = t
        self.TB = tuple([0] + [x+1 for x in t])
        r = [0] * len(t)
        for i, tt in enumerate(t):
            r[tt] = i
        self.RB= tuple([0] + [x+1 for x in r])
        self.OutShape = tuple([self.InShape[j] for j in self.T])

    def fprop(self, x, state_in):
        return x[0].transpose(self.TB), None
        
    def bprop(self, x, state_in, y, state_out, gY, gState):
        return [gY.transpose(self.RB)], None, [] 
        
class Flatten(Layer):
    def __init__(self, inp, name = None):
        Layer.__init__(self, [inp], name)
        self.InShape = inp.shape
        self.OutShape = (np.prod(inp.shape),)
    
    def fprop(self, x, in_state):
        x = x[0]
        mb = x.shape[0]
        y = np.reshape(x, (mb, -1))
        #print "Flatten: x.shape=", x.shape, "   y.shape=",y.shape
        
        return y, None

    def bprop(self, x, state_in, y, state_out, gY, gState):
        gx = np.reshape(gY, x[0].shape)        
        return [gx], None, []

class Scale(Layer):
    
    def __init__(self, inp, a, b, name=None):
        # linear scale layer: y = a*x + b where a and b are constants 
        # or y = np.dot(x, a) + b
        # If A is a matrix, it has to have shape of ()
        Layer.__init__(self, [inp], name)
        assert inp.shape[-1] == a.shape[-2]
        assert len(a.shape) == 2
        self.OutShape = tuple(list(inp.shape)[:-1] + [a.shape[-1]])
        assert self.OutShape == b.shape

    def fprop(self, x, in_state):
        return x[0].dot(self.A) + self.B, None
        
    def bprop(self, x, state_in, y, state_out, gY, gState):
        gx = np.dot(self.A, gY.T).T      # [nx, ny] dot [ny, nb] -> [nx, nb] -> [nb, nx]
        return [gx], None, []
        
class Bias(Layer):
    
    def __init__(self, inp, name=None):
        # linear scale layer: y = a*x + b where a and b are constants 
        # or y = np.dot(x, a) + b
        # If A is a matrix, it has to have shape of ()
        Layer.__init__(self, [inp], name)
        self.OutShape = inp.shape
        
    def initParams(self, rng=None):
        self.B = np.zeros(self.OutShape)        

    def fprop(self, x, state_in):
        return x[0] + self.B, None

    def bprop(self, x, state_in, y, state_out, gY, gState):
        return [gY], None, [np.mean(gY, axis=0)]
        
    def setParams(self, params):
        self.B = params[0]
        
    def getParams(self):
        return (self.B,)
        
    def getParamsAsDict(self):
        return {"B": self.B}
        
    def setParamsAsDict(self, dct):
        self.B = dct["B"]
        
class Linear(Layer):
    def __init__(self, inp, nout, name = None, applier = None, weight_decay=1.e-5):
        Layer.__init__(self, [inp], name)
        #print "input shape=", inp.shape
        self.InShape = inp.shape
        self.n_in = self.InShape[-1]     
        self.n_out = nout
        self.OutShape = self.InShape[:-1]+(nout,)
        #print "out shape:", out_shape
        self.weight_decay = weight_decay
        self.Applier = applier

    def initParams(self, rng=None):
        #print "Linear: input_shape=", input_shape
        if rng == None: rng = np.random.RandomState()
        W_shape = (self.n_in, self.n_out)
        #print "W_shape=", W_shape
        self.W = rng.normal(size=W_shape, scale=1.0/math.sqrt(self.n_in))
        self.b = np.zeros(self.n_out)

    def fprop(self, x, in_state):
        out = np.dot(x[0], self.W) + self.b
        return out, None

    def bprop(self, x, state_in, y, state_out, gY, gState):
        #print "linear.bprop: x:", x[0].shape, "   gy:", gY.shape
        x = x[0]
        n_mb = len(x)
        inp_flat = x.reshape((-1, self.n_in))
        g_flat = gY.reshape((-1, self.n_out))
        gW = np.dot(inp_flat.T, g_flat)/n_mb    # [...,n_in],[...,n_out]
        gb = np.mean(g_flat, axis=0)
        gx = np.dot(gY, self.W.T)
        #print "Linear.bprop: gx:", gx.shape
        return [gx], None, [gW, gb]

    def regularize(self):
        self.W *= (1.0-self.weight_decay)

    def getParams(self):
        return self.W, self.b

    def setParams(self, tup):
        self.W, self.b = tup

    def getParamsAsDict(self):     
        return {"W": self.W.copy(), "b":self.b.copy()}
        
    def setParamsAsDict(self, dct):      
        self.W = dct["W"].copy()
        self.b = dct["b"].copy()
        
    def pruneStats(self):
        out = []
        cor = np.corrcoef(self.W, rowvar=0)
        for i in range(self.n_out-1):
            for j in range(i+1, self.n_out):
                out.append(cor[i,j])
        return np.abs(np.array(out))

class LossMixin(object):
    def total_loss(self, y_, y):
        """ Calculate total, unnormalized loss given output and predicted output. """
        raise NotImplementedError()

    def input_grad(self, output, output_pred):
        """ Calculate input gradient given output and predicted output. """
        raise NotImplementedError()
        
    def loss(self, y_, y):
        assert len(y_) == len(y)
        return self.total_loss(y_, y)/len(y_)
        
    average_loss = loss     # average loss over the minibatch
    
    display_loss = loss
        
class L2Regression(Layer, LossMixin):
    
    def __init__(self, inp, name=None, error_scale=1.0):
        Layer.__init__(self, [inp], name)
        self.ErrorScale = error_scale
    
    def fprop(self, x, in_state):
        return x[0], None

    def bprop(self, gY):
        raise NotImplementedError(
            'L2Regression does not support back-propagation of gradients. '
            + 'It should occur only as the last layer of a NeuralNetwork.'
        )

    def input_grad(self, y_, y):
        mb = y.shape[0]
        self.gX = (y - y_)
        return self.gX

    def total_loss(self, y_, y):
        mb = y.shape[0]
        return np.sum(np.square(y_-y))/2
        
    def error_(self, y_, y):
        diff = np.abs(y_ - y)
        n = np.sum(diff < 0.5)
        return 1.0 - float(n)/y_.size

    def error(self, y_, y):
        return math.sqrt(np.mean(np.square(y_-y)))*self.ErrorScale

    def error_(self, y_, y):
        cov = np.corrcoef(y_.reshape(-1), y.reshape(-1))
        return cov[0,1]

class LogRegression(Layer, LossMixin):
    """ Softmax layer with cross-entropy loss function. """


    def __init__(self, inp, name=None):
        Layer.__init__(self, [inp], name)
    


    def fprop(self, inp, in_state):
        #print "LogRegression: input:", inp.shape
        inp = inp[0]
        e = np.exp(inp - np.amax(inp, axis=1, keepdims=True))
        return e/np.sum(e, axis=1, keepdims=True), None

    def bprop(self, gY):
        raise NotImplementedError(
            'LogRegression does not support back-propagation of gradients. '
            + 'It should occur only as the last layer of a NeuralNetwork.'
        )

    def input_grad(self, y_, y):
        # Assumes one-hot encoding.
        self.gX = (y - y_)      #/y_.shape[1]
        return self.gX

    def total_loss(self, y_, y):
        # Assumes one-hot encoding.
        eps = 1e-15
        y = np.clip(y, eps, 1 - eps)
        ynorm = y/y.sum(axis=1, keepdims=True)
        #print Y_pred, np.sum(Y_pred, axis=1)
        loss = -np.sum(y_ * np.log(ynorm))
        return loss 
        
    def display_loss(self, y_, y):
        #print "display_loss"
        n = math.exp(-self.average_loss(y_, y))
        return n
        
    def error(self, y_, y):
        #print np.sum(np.argmax(y_, axis=1) == np.argmax(y, axis=1))
        return 1.0-float(np.sum(np.argmax(y_, axis=1) == np.argmax(y, axis=1)))/y_.shape[0]

class SaveModelMixin:
    
    def dump(self, key_root):
        d = {}
        for i, layer in enumerate(self.layers):
            if layer.params:
                key = "layer_%d" % (i,)
                prefix = key + "/"
                params = layer.paramsAsDict
                d.update(
                    dict(((prefix+k, v) for k, v in params.items()))
                )
        return d
            
    def restore(self, dump):
        for i, layer in enumerate(self.layers):
            if layer.params:
                key = "layer_%d" % (i,)
                prefix = key + "/"
                params = dict(
                    ((k-prefix, v) for k, v in dump.items() if k.beginswith(prefix))
                )
                layer.paramsAsDict = params
        
    def save(self, fn):
        dump = self.dump("")
        np.savez(fn, **dump)
        #print "Network saved to %s" % (fn,)
        
    def load(self, fn):
        dump = np.load(fn)
        self.restore(dump)
        #print "Network loaded from %s" % (fn,)

