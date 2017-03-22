from mynnet import Layer
from .helpers import sigmoid, sigmoid_d, relu, relu_d, tanh, tanh_d
import numpy as np

class Activation(Layer):
    
    def __init__(self, inp, name = None):
        Layer.__init__(self, [inp], name)
        self.OutShape = inp.shape

    def fun(self, x):
        raise ValueError('Method fun() not implemented.')

    def fun_d(self, x, y):
        raise ValueError('Method fun_d() not implemented.')

    def fprop(self, x, in_state):
        y = self.fun(x[0])
        #print "%s: computed" % (self,)
        return y, None

    def bprop(self, x, state_in, y, state_out, gY, gState):
        #print "Activation: x:%s, y:%s" % (x, y)
        g = self.fun_d(x[0], y)
        gX = gY*g
        #print "Activation: gY:%s,  g:%s" % (gY.shape, g.shape)
        return [gX], None, []

class Sigmoid(Activation):
    
    def fun(self, x):
        return sigmoid(x)

    def fun_d(self, x, y):
        return y*(1.0-y)
        
class ReLU(Activation):
    
    def fun(self, x):
        return relu(x)

    def fun_d(self, x, y):
        return relu_d(x)
        
class Tanh(Activation):
    
    def fun(self, x):
        #print np, np.tanh(x)
        y = np.tanh(x)
        return y
        
    def fun_d(self, x, y):
        return 1.0 - np.square(y)
        
class SoftPlus(Activation):
    
    def fun(self, x):
        return np.log(np.exp(x)+1.0)

    def fun_d(self, x, y):
        e = np.exp(x)
        return e/(1.0+e)

        
   

