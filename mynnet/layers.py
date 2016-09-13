import numpy as np
import math, time

from convnet.conv import convolve_xw, convolve_xy
from convnet.pool import pool, bprop_pool

class Layer(object):
    
    def __init__(self, inp, name=None):
        self.Name = name
        self.Input = inp
        if self.Input:
            self.InShape = self.Input.out_shape()
        self.VisitId = None
        self.StepId = None
            
    def __str__(self):
        return "<Layer %s>" % (self.Name or self.__class__)
    
    def fprop(self, input):
        """ Calculate layer output for given input (forward propagation). """
        raise NotImplementedError()

    def bprop(self, output_grad):
        """ Calculate input gradient. """
        raise NotImplementedError()

    def output_shape(self, input_shape):
        return input_shape
        
    #
    # Functional API, default implementation
    #
    
    def init(self, rng):
        self.init_layer(rng)
        if self.Input:
            self.Input.init(rng)
    
    def inputs(self):
        return [] if self.Input is None else [self.Input]
            
    def init_layer(self, rng):
        pass
        
    def grad_mag(self, g):
        g_shape = g.shape
        nb = g_shape[0]
        g_flat = g.reshape((nb, -1))
        g_sq = np.sqrt(np.sum(np.square(g_flat), axis=1))
        return np.mean(g_sq)
        
        
    def backprop_rec(self, gy, normalize_grads = False):
        #print "%s: backprop.." % (self,)
        gx = self.bprop(gy)
        if normalize_grads:
            gx_shape = gx.shape
            gx_flat = gx.reshape((gx_shape[0], -1))
            gy_sq = self.grad_mag(gy)
            gx_sq = self.grad_mag(gx)
            #f = np.expand_dims(gy_sq/gx_sq, 1)
            f = gy_sq/gx_sq
            gx = (gx_flat*f).reshape(gx_shape)
            #print str(self), "  gx initial:", np.mean(gx_sq), "  gy:", np.mean(gy_sq), "  gx_final:", np.mean(self.grad_mag(gx))
            #print str(self), "  gx initial:", gx_sq, "  gy:", gy_sq, "  gx_final:", self.grad_mag(gx)
        else:
            #print str(self), "  gx:", self.grad_mag(gx), "  gy:", self.grad_mag(gy)
            pass
        return self.Input.backprop_rec(gx, normalize_grads = normalize_grads)
        
    def updateParams(self, eta):
        pass        # applicable only for ParamMixin. Will override there
        
    def train_rec(self, eta, step_id = None):

        if step_id == None: step_id = time.time()

        if step_id != self.StepId:
            self.StepId = step_id
            self.updateParams(eta)
            
        if self.Input:
            self.Input.train_rec(eta, step_id)

    def reset_grads_rec(self):
        #print "%s: reset_grads_rec" % (self,)
        self.reset_grads()
        if self.Input:
            self.Input.reset_grads_rec()

    def reset_compute_rec(self):
        self.reset_compute()
        if self.Input:
            self.Input.reset_compute_rec()
            
    def reset_compute(self):
        self.Y = None

    def reset_grads(self):
        pass
            
    def compute(self):
        if self.Y is None:
            self.Y = self.fprop(self.Input.compute())
        return self.Y
        
    def out_shape(self):
        return self.output_shape(self.Input.out_shape())
        
    @property
    def shape(self):
        return self.out_shape()

    def recursive(self, callback, carry, args, visit_id = None):
        if not visit_id is None and visit_id == self.VisitId: return
        carry = callback(self, carry, args)
        self.VisitId = visit_id
        for i in self.inputs():
            carry = i.recursive(callback, carry, args, visit_id)

class Concat(Layer):
    
    # concatenates along specified axis (mb, axis0, axis1, axis2,...)
    # assumes all other dimensions except "axis" are the same
    # remember that input.output_shape() returns shape without the minibatch dimension
    
    def __init__(self, inputs, axis=0, name=None):
        Layer.__init__(self, None, name)
        self.Axis = axis            # axis without including the minibatch dimension
        self.Inputs = inputs
        self.InShapes = [i.out_shape() for i in self.Inputs]
        self.InShape = tuple(self.InShapes)     # for the base class
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
        
    def init(self, rng):
        if self.Inputs:
            for i in self.Inputs:
                i.init(rng)
                
    def inputs(self):
        return self.Inputs

    def compute(self):
        # concatenation here occurs over axis=1, which is fist after the batch dimension
        ys = [i.compute() for i in self.Inputs]
        #print [yy.shape for yy in ys]
        self.Y = np.concatenate(ys, axis=self.Axis+1)                # +1 for the minibatch
        #print y.shape
        return self.Y
    
    def backprop_rec(self, gy, normalize_grads = False):
        gy_split = np.split(gy, self.Splits, axis=self.Axis+1)      # +1 for the minibatch
        gx = [l.backprop_rec(gyi, normalize_grads = normalize_grads) for l, gyi in zip(self.Inputs, gy_split)]
        self.gY = gy
        self.gX = gx
        return gx

    def reset_grads_rec(self):
        for i in self.Inputs:
            i.reset_grads_rec()

    def reset_compute_rec(self):
        for i in self.Inputs:
            i.reset_compute_rec()

    def train_rec(self, eta, step_id = None):
        for i in self.Inputs:
            i.train_rec(eta, step_id)

    def out_shape(self):
        return self.OutShape
        
class Reshape(Layer):
    
    def __init__(self, inp, out_shape, name=None):
        Layer.__init__(self, inp, name)
        assert np.prod(self.InShape) == np.prod(out_shape)
        self.OutShape = out_shape

    def fprop(self, x):
        self.X = x
        return x.reshape((-1,)+self.OutShape)
        
    def bprop(self, gy):
        self.gY = gy
        self.gX = gy.reshape(self.X.shape)
        return self.gX
        
    def out_shape(self):
        return self.OutShape
        
class InputLayer(Layer):
    
    def __init__(self, shape, name=None):
        Layer.__init__(self, None, name)
        # shape without minibatch dim
        #print "shape=", shape
        self.InShape = shape
        self.OutShape = shape
        self.reset_grads()
    
    def set(self, x):
        assert x.shape[1:] == self.InShape, "Invalid input for input layer %s: %s, expected %s" % (
            self, x.shape[1:], self.InShape)
        self.X = x
        
    def compute(self):
        return self.X
        
    def backprop_rec(self, gy, normalize_grads = False):
        return self.bprop(gy)
        
    def reset_grads(self):
        #print "reset grads"
        self.gX = None
        
    def bprop(self, gy):
        if self.gX is None:
            #print "set gx to", gy
            self.gX = gy.copy()/len(gy)
        else:
            #print "adding gy", gy
            self.gX += gy/len(gy)
        self.gY = self.gX
        return self.gX
        
    def out_shape(self):
        return self.OutShape
        
class LossMixin(object):
    def loss(self, output, output_pred):
        """ Calculate mean loss given output and predicted output. """
        raise NotImplementedError()

    def input_grad(self, output, output_pred):
        """ Calculate input gradient given output and predicted output. """
        raise NotImplementedError()


class ParamMixin(object):
    def __init__(self, applier):
        self.Applier = applier
        
    def set_applier(self, applier):
        self.Applier = applier
        
    def __get_params(self):
        """ Layer parameters. """
        raise NotImplementedError()
    
    def __set_params(self):
        """ Layer parameters. """
        raise NotImplementedError()
        
    def param_grads(self):
        """ Get layer parameter gradients as calculated from bprop(). """
        raise NotImplementedError()

    def param_incs(self):
        """ Get layer parameter steps as calculated from bprop(). """
        raise NotImplementedError()
        
    def dump_params(self, prefix):
        """ Export parameters into a dictionary with given key prefix """
        raise NotImplementedError()
        
    def restore_params(self, prefix, dct):
        """ Load parameters from a dictionary with given key prefix """
        raise NotImplementedError()
        
    def init_layer(self, rng):
        raise NotImplementedError()

    def gradsToDeltas(self, eta, *grads):
        
        if eta == 0.0:  # no updates
            return tuple([np.zeros(g.shape) for g in grads])
        
        applier = self.Applier
        if applier:
            deltas = applier(eta, *grads)
        else:
            #print "%s: no applier" % (self,)
            deltas = tuple([-eta*g for g in grads])
        return deltas
        
    def addDeltas(self, *deltas):
        """ add deltas to the parameters, must be implemented by the concrete class """
        raise NotImplementedError()

    def regularize(self):
        """ regularize parameters, must be implemented by the concrete class """
        raise NotImplementedError()
        
    def reset_grads(self):
        """ reset parameter gradients to zeros, must be implemented by the concrete class """
        raise NotImplementedError()

    def updateParams(self, eta):
        if eta:
            deltas = self.gradsToDeltas(eta, *self.param_grads())
            self.addDeltas(*deltas)
            self.regularize()
        

class Transpose(Layer):
    
    def __init__(self, inp, t, name=None):
        Layer.__init__(self, inp, name)
        self.T = t
        self.TB = tuple([0] + [x+1 for x in t])
        r = [0] * len(t)
        for i, tt in enumerate(t):
            r[tt] = i
        self.RB= tuple([0] + [x+1 for x in r])

    def fprop(self, x):
        return x.transpose(self.TB)
        
    def bprop(self, gy):
        self.gY = gy
        self.gX = gy.transpose(self.RB)    
        return self.gX
        
    def out_shape(self):
        #print "input shape:", self.InShape
        #print "T:", self.T
        t = tuple([self.InShape[j] for j in self.T])
        #print "trasposed to:", t
        return t
        
class Linear(ParamMixin, Layer):
    def __init__(self, inp, n_out, name = None, applier = None, weight_decay=1.e-5):
        Layer.__init__(self, inp, name)
        ParamMixin.__init__(self, applier)
        assert len(self.InShape) == 1   
        self.n_in = self.InShape[0]     
        self.n_out = n_out
        self.weight_decay = weight_decay
        self.gW = None

    def init_layer(self, rng):
        #print "Linear: input_shape=", input_shape
        W_shape = (self.n_in, self.n_out)
        self.W = rng.normal(size=W_shape, scale=1.0/math.sqrt(self.n_in))
        self.b = np.zeros(self.n_out)

    def fprop(self, input):
        self.last_input = input
        #print "in shape:", input.shape
        #print "W shape:", self.W.shape
        #print "b shape:", self.b.shape
        return np.dot(input, self.W) + self.b

    def reset_grads(self):
        self.gW = None
        self.gb = None
        self.gX = None
        self.gY = None

    def bprop(self, output_grad):
        n_mb = self.last_input.shape[0]
        gW = np.dot(self.last_input.T, output_grad)/n_mb
        gb = np.sum(output_grad, axis=0)/n_mb
        out = np.dot(output_grad, self.W.T)
        
        if self.gW is None:
            self.gW = gW
            self.gb = gb
            self.gX = out
            self.gY = output_grad.copy()
        else:
            self.gW += gW
            self.gb += gb
            self.gX += out
            self.gY += output_grad        
        return out

    def addDeltas(self, dw, db):
        self.W += dw
        self.b += db
        
    def regularize(self):
        self.W *= (1.0-self.weight_decay)

    def __get_params(self):
        return self.W, self.b

    def __set_params(self, tup):
        self.W, self.b = tup

    params = property(__get_params, __set_params)
    
    def param_incs(self):
        return self.dW, self.db

    def param_grads(self):
        return self.gW, self.gb

    def output_shape(self, input_shape):
        return (self.n_out,)

    def dump_params(self, prefix):
        return {
            prefix+"/W":    self.W,
            prefix+"/B":    self.b
        }

    def restore_params(self, prefix, dct):
        self.W = dct[prefix+"/W"]
        self.b = dct[prefix+"/B"]

class Flatten(Layer):
    
    def fprop(self, inp):
        self.X = inp
        return np.reshape(self.X, (self.X.shape[0], -1))

    def bprop(self, output_grad):
        #print "Flatten.bprop: X:", self.X.shape, "  output:",output_grad.shape
        self.gY = output_grad
        self.gX = np.reshape(output_grad, self.X.shape)        
        return self.gX

    def output_shape(self, input_shape):
        return (np.prod(input_shape),)


class L2Regression(Layer, LossMixin):
    
    def fprop(self, input):
        return input

    def bprop(self, output_grad):
        raise NotImplementedError(
            'L2Regression does not support back-propagation of gradients. '
            + 'It should occur only as the last layer of a NeuralNetwork.'
        )

    def input_grad(self, y_, y):
        self.gX = y - y_
        return self.gX

    def loss(self, y_, y):
        mb = y.shape[0]
        return np.sum(np.square(y_-y))/2/mb

    def error(self, y_, y):
        diff = np.abs(y_ - y)
        n = np.sum(diff < 0.5)
        return 1.0 - float(n)/y_.size

    def error_(self, y_, y):
        return math.sqrt(np.mean(np.square(y_-y)))

    def error_(self, y_, y):
        cov = np.corrcoef(y_.reshape(-1), y.reshape(-1))
        return cov[0,1]

class LogRegression(Layer, LossMixin):
    """ Softmax layer with cross-entropy loss function. """
    def fprop(self, input):
        
        e = np.exp(input - np.amax(input, axis=1, keepdims=True))
        return e/np.sum(e, axis=1, keepdims=True)

    def bprop(self, output_grad):
        raise NotImplementedError(
            'LogRegression does not support back-propagation of gradients. '
            + 'It should occur only as the last layer of a NeuralNetwork.'
        )

    def input_grad(self, y_, y):
        # Assumes one-hot encoding.
        self.gX = (y - y_)/y_.shape[1]
        return self.gX

    def loss(self, Y, Y_pred):
        # Assumes one-hot encoding.
        eps = 1e-15
        Y_pred = np.clip(Y_pred, eps, 1 - eps)
        ynorm = Y_pred/Y_pred.sum(axis=1, keepdims=True)
        #print Y_pred, np.sum(Y_pred, axis=1)
        loss = -np.mean(Y * np.log(ynorm))
        return loss 
        
    def error(self, y_, y):
        #print np.sum(np.argmax(y_, axis=1) == np.argmax(y, axis=1))
        return 1.0-float(np.sum(np.argmax(y_, axis=1) == np.argmax(y, axis=1)))/y_.shape[0]

class SaveModelMixin:
    
    def dump(self, key_root):
        d = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, ParamMixin):
                key = "layer_%d" % (i,)
                d.update(layer.dump_params(key))
        return d
            
    def restore(self, dump):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, ParamMixin):
                key = "layer_%d" % (i,)
                layer.restore_params(key, dump)
        
    def save(self, fn):
        dump = self.dump("")
        np.savez(fn, **dump)
        #print "Network saved to %s" % (fn,)
        
    def load(self, fn):
        dump = np.load(fn)
        self.restore(dump)
        #print "Network loaded from %s" % (fn,)

