from .layers import Layer
from .helpers import sigmoid, sigmoid_d, relu, relu_d, tanh, tanh_d
import numpy as np

class Activation(Layer):

    def fun(self, x):
        raise ValueError('Method fun() not implemented.')

    def fun_d(self, x, y):
        raise ValueError('Method fun_d() not implemented.')

    def fprop(self, input):
        self.X = input
        #self.last_input = input
        self.Y = self.fun(self.X)
        #print "%s: computed" % (self,)
        return self.Y

    def bprop(self, output_grad):
        #print self
        #print "output_grad:", output_grad.shape
        #print "X shape=", self.X.shape
        #print "Y shape=", self.Y.shape
        g = self.fun_d(self.X, self.Y)
        #print "g shape:", g.shape
        xgrad = output_grad*g
        #print "Activation(%s):   y_grad=%s    x_grad=%s" % (self, output_grad[0], xgrad[0])
        return xgrad

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
        return np.tanh(x)

    def fun_d(self, x, y):
        return 1.0 - np.square(y)
        
class SoftPlus(Activation):
    
    def fun(self, x):
        return np.log(np.exp(x)+1.0)

    def fun_d(self, x, y):
        e = np.exp(x)
        return e/(1.0+e)

        
   

