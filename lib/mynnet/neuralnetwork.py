import numpy as np
from scipy.optimize import check_grad
from .layers import ParamMixin, SaveModelMixin
from .helpers import one_hot, unhot


class NeuralNetwork(SaveModelMixin):
    
    def __init__(self, input_shape, layers, rng=None):
        print "NeuralNetwork.__init__()"
        self.layers = layers
        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng
        self.init(input_shape)
        
    def init(self, input_shape):
        # input_shape is (batch, ...)
        # Setup layers sequentially
        next_shape = input_shape
        for layer in self.layers:
            next_shape = layer._setup(next_shape, self.rng)
            print layer, next_shape
        #if next_shape != Y.shape:
        #    raise ValueError('Output shape %s does not match Y %s'
        #                     % (next_shape, Y.shape))

    def hello(self, qq):
        print qq
        
    def compute(self, X):
        self.X = X
        x = X
        #print "compute"
        for layer in self.layers:
            x = layer.fprop(x)
            #print layer, "x->", x[0]
        self.Y = x
        return self.Y
            
    def train(self, X, Y, eta):
        y = self.compute(X)
        
        grad = self.layers[-1].input_grad(Y, y)
        
        #print "train: X=", X[0], "  y=", y[0], "  y_=", Y[0], "  grad=", grad[0]
        
        for layer in reversed(self.layers[:-1]):
            grad = layer.bprop(grad, eta)
            #print layer, "grad->", grad[0]
            
        #print "y new=", self.compute(X)[0]
        #print
        self.GX = grad
        return y
        
        
    def fit(self, X, Y, learning_rate=0.1, max_iter=10, batch_size=64):
        """ Train network on the given data. """
        #print "Y=", Y[0]
        
        #print Y[:10]
        
        n_samples = Y.shape[0]
        n_batches = n_samples // batch_size
        Y_one_hot = one_hot(Y)
        #print "Y_one_hot=", Y_one_hot[0]
        iter = 0
        # Stochastic gradient descent with mini-batches
        while iter < max_iter:
            iter += 1
            for b in range(n_batches):
                batch_begin = b*batch_size
                batch_end = batch_begin+batch_size
                X_batch = X[batch_begin:batch_end]
                Y_batch = Y_one_hot[batch_begin:batch_end]

                # Forward propagation
                Y_pred = self.compute(X_batch)
                #if b == 0:
                #    #print "Y_pred=",Y_pred[0]

                # Back propagation of partial derivatives
                next_grad = self.layers[-1].input_grad(Y_batch, Y_pred)
                for layer in reversed(self.layers[:-1]):
                    next_grad = layer.bprop(next_grad, learning_rate)

                # Update parameters
                #for layer in self.layers:
                #    if isinstance(layer, ParamMixin):
                #        for param, inc in zip(layer.params(),
                #                              layer.param_incs()):
                #            param -= learning_rate*inc

            # Output training status
            Y_pred = self.compute(X)
            loss = self.loss(Y_one_hot)
            error = self.error(Y)
            #for i in range(10):
            #    print Y[i], self.Y[i]
            print('iter %i, loss %.4f, train error %.4f' % (iter, loss, error))

    def loss(self, Y):
        return self.layers[-1].loss(Y, self.Y)
        
    def norm_loss(self, Y):
        return self.layers[-1].norm_loss(Y, self.Y)
        
    def error(self, Y):
        """ Calculate error on the given data. """
        Y_pred = unhot(self.Y)
        error = Y_pred != Y
        return np.mean(error)

    def check_gradients(self, X, Y):
        """ Helper function to test the parameter gradients for
        correctness. """
        self.train(X, Y, 0.0)       # force grad calculation, but do not actually update any weights
        L0 = self.loss(Y)
        dp = 0.001
        for l, layer in enumerate(reversed(self.layers[:-1])):
            if isinstance(layer, ParamMixin):
                print('layer #%d %s' % (l, layer))
                for p, (param, paramg) in enumerate(zip(layer.params(), layer.param_grads())):
                    param_shape = param.shape
                    pflat = param.flat
                    #print pflat
                    pgflat = paramg.flat
                    for i in range(param.size):

                        v0 = pflat[i]
                        pflat[i] = v0-dp
                        self.compute(X)
                        lminus = self.loss(Y)

                        pflat[i] = v0+dp
                        self.compute(X)
                        lplus = self.loss(Y)

                        pflat[i] = v0
                        
                        grad_c = (lplus-lminus)/(dp*2)
                        grad_a = pgflat[i]
                        
                        print "param %d, index %d: grad_c:%f, grad_a:%f, delta:%f" % (p, i, grad_c, grad_a, grad_a-grad_c)

        for i in range(X.size):
            gradx_a = self.GX.flat[i]

            x0 = X.flat[i]
            
            X.flat[i] = x0 + dp
            self.compute(X)
            lplus = self.loss(Y)

            X.flat[i] = x0 - dp
            self.compute(X)
            lminus = self.loss(Y)
            
            gradx_c = (lplus-lminus)/(dp*2)
            X.flat[i] = x0
            
            print "x[%d] gradx_c:%f, gradx_a:%f, delta:%f" % (i, gradx_c, gradx_a, grad_a-grad_c)

        self.compute(X)
            
                        
                    
