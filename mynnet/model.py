import numpy as np
import random, math
from mynnet import L2Regression, ParamMixin, SaveModelMixin

class ParamSet:
    def __init__(self, lst):
        self.Params = lst       # list of tuples: [(W,b), (W,b), (W,V,b), ...]
        
    @property
    def params(self):
        return self.Params
        
    def copy(self, ps, deep=False):
        if deep:
            lst = []
            for tup in self.params:
                lst.append(tuple([p.copy() for p in tup]))
        else:
            lst = p.params
        return ParamSet(lst)
        
    def randomBias(self, alpha):
        # generate random parameters in range of alpha * size of the params
        deltas = []
        for lp in self.Params:
            params = []
            for p in lp:
                s = math.sqrt(np.mean(np.square(p)))*alpha
                #print p.shape
                params.append(s*np.random.randn(*p.shape))
            deltas.append(tuple(params))
        return ParamSet(deltas)
        
    def __add__(self, ps):
        lst = []
        for p, q in zip(self.params, ps.params):
            lst.append(tuple([x+y for x, y in zip(p, q)]))
        return ParamSet(lst)
        
    def __sub__(self, ps):
        lst = []
        for p, q in zip(self.params, ps.params):
            lst.append(tuple([x-y for x, y in zip(p, q)]))
        return ParamSet(lst)
        
    def __mul__(self, s):
        lst = []
        for p in self.params:
            lst.append(tuple([x*s for x in p]))
        return ParamSet(lst)

    def __div__(self, x):
        return self * (1.0/x)
        

class Model(SaveModelMixin):
    
    def __init__(self, inputs, out, loss=None, rng=None, applier_class=None, applier_params=()):
        #print "NeuralNetwork.__init__()"
        if not type(inputs) in (type([]), type(())):
            inputs = [inputs]
        self.Inputs = inputs
        self.Out = out
        self.Loss = loss or L2Regression(self.Out)
        if rng is None:
            rng = np.random.RandomState()
        self.Loss.init(rng)
        
        if not applier_class is None:
            for l in self.layers:
                if isinstance(l, ParamMixin):
                    a = applier_class(*applier_params)
                    l.set_applier(a)
        
    def compute(self, x):
        if not type(x) in (type([]), type(())):
            x = [x]
        self.Loss.reset_compute_rec()
        self.X = x
        for ii, xx in zip(self.Inputs, x):
            ii.set(xx)
        self.Y = self.Loss.compute()
        return self.Y
        
    __call__ = compute
            
    def train(self, x, y_, eta, normalize_grads = False):
        #print "mode.train"
        y = self.compute(x)
        loss = self.Loss.loss(y_, y)
        error = self.Loss.error(y_, y)
        
        grad = self.Loss.input_grad(y_, y)
        self.GY = grad
        
        #print "train: grad=", grad.shape
        self.Out.reset_grads_rec()
        self.Out.backprop_rec(grad, normalize_grads=normalize_grads)
        gx = [inp.gX for inp in self.Inputs]
        self.GX = gx
        self.Out.train_rec(eta)
        return y, loss, error
        
    def loss(self, y_, x=None):
        if not x is None:
            self.compute(x)
        return self.Loss.loss(y_, self.Y)
        
    def error(self, y_, x=None):
        if not x is None:
            self.compute(x)
        return self.Loss.error(y_, self.Y)
        
    @property
    def layers(self):
        to_scan = [self.Loss]
        all_layers = []
        while to_scan:
            new_scan = []
            for l in to_scan:
                if not l in all_layers:
                    all_layers.append(l)
                    new_scan += l.inputs()
            to_scan = new_scan
        return all_layers
        
    def __getitem__(self, name):
        # finds layer by name
        for l in self.layers:
            if l.Name == name:  return l
        raise KeyError, "Layer %s not found" % (name,)
        
    def __get_params(self):
        return ParamSet([l.params for l in self.layers if isinstance(l, ParamMixin)])
        
    def __set_params(self, ps):
        #print "Model.set_params"
        lst = [l for l in self.layers if isinstance(l, ParamMixin)]
        for l, p in zip(lst, ps.params):
            l.params = p
            
    params = property(__get_params, __set_params)

    def nparams(self):
        nparams = 0
        for l in self.layers:
            if isinstance(l, ParamMixin):
                for p in l.params:
                    nparams += p.size
        return nparams
    
        
    def check_gradients(self, X = None, Y = None, check_params = True, check_inputs = True, mb_size=2,
                    random_params = None, random_inputs = None):
        """ Helper function to test the parameter gradients for
        correctness. """
        
        
        if X == None:
            X = [np.random.random((mb_size,) + inp.InShape) for inp in self.Inputs]
        if Y == None:
            xr = [np.random.random((mb_size,) + inp.InShape) for inp in self.Inputs]
            Y = self.compute(xr)
            
        
        self.train(X, Y, 0.0)       # force grad calculation, but do not actually update any weights
        L0 = self.loss(Y)
        
        if check_params:
            dp = 0.001
            for l in self.layers:
                if isinstance(l, ParamMixin):
                    nparams = sum([p.size for p in l.params])
                    print "Checking layer %s with %d parameters" % (l, nparams)
                    for p, (param, paramg) in enumerate(zip(l.params, l.param_grads())):
                        param_shape = param.shape
                        pflat = param.flat
                        #print pflat
                        pgflat = paramg.flat
                        rng = range(param.size) 
                        if not random_params is None and param.size > random_params:
                            rng = random.sample(rng, random_params)
                        for i in rng:

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
                            delta = grad_a-grad_c
                            if grad_c != grad_a and (abs(delta) > 1.e-4 or abs(grad_c/grad_a-1) > 1.0e-2):
                                print "param %d, index %d: grad_c:%s, grad_a:%s, delta:%s, grad_c/grad_a=%f" % (p, i, grad_c, grad_a, delta, grad_c/grad_a)

        if check_inputs:
            print "Checking inputs"
        
            dx = 0.0001
        
            for j, (x, gx_a) in enumerate(zip(self.X, self.GX)):
                    rng = range(x.size) 
                    if not random_inputs is None and x.size > random_inputs:
                        rng = random.sample(rng, random_inputs)
                    for i in rng:
                
                        gradx_a = gx_a.flat[i]

                        x0 = x.flat[i]
            
                        x.flat[i] = x0 + dx
                        self.compute(X)
                        lplus = self.loss(Y)

                        x.flat[i] = x0 - dx
                        self.compute(X)
                        lminus = self.loss(Y)
            
                        gradx_c = (lplus-lminus)/(dx*2)
                        x.flat[i] = x0
                        delta = gradx_a-gradx_c
                        if gradx_a != gradx_c and (abs(delta) > 1.0e-4 or abs(gradx_a/gradx_c-1) > 1.0e-2):
                            print "x[%d] gradx_c:%s, gradx_a:%s, delta:%s, %f" % (i, gradx_c, gradx_a, gradx_a-gradx_c, gradx_a/gradx_c)

        self.compute(X)
        
    @staticmethod
    def trainSimplex(nets, batches):
        # assumes nets is a list of at least 3 Model instances with identical structure
    
        # check parameters
        assert len(nets) >= 3

        losses = []
    
        for bx, by in batches:
            losses = []
            for n in nets:
                losses.append((n.loss(by, x=bx), n))
            losses.sort()       # sort by the loss, ascending
            bv, bn = losses[0]  # best loss and best network
            wv, wn = losses[-1] # the worst
            wp = wn.params
            bp = bn.params
            
            #print "losses:", [l for l, n in losses]
        
            # permutate randomly to break away from the hyperplane of lower dimesions
            if random.random() < 0.01:
                dp = bp - wp
                bp = bp + dp.randomBias(0.5)
                bn.params = bp
                bv = bn.loss(by, x=bx)
                losses[0] = (bv, bn)
                if bv > losses[1][0]:
                    losses.sort()
                    bv, bn = losses[0]  # best loss and best network
                    wv, wn = losses[-1] # the worst
                    wp = wn.params
                    bp = bn.params
                    
                
        
            # calculate the center
            cp = bn.params
            for v, n in losses[1:-1]:
                cp = cp + n.params
            cp = cp /(len(losses)-1)
        
            # reflection
            rp = cp + (cp-wn.params)
            #print "wn.params <- rp"
            wn.params = rp
            rv = wn.loss(by, x=bx)
        
            if rv < bv:
                # expansion
                ep = rp + (rp - cp)
                #print "wn.params <- ep"
                wn.params = ep
                ev = wn.loss(by, x=bx)
            
                if ev < bv:
                    pass        # wn.params = ep already done
                else:
                    #print "wn.params <- rp"
                    wn.params = rp
            else:
                if rv < losses[-2][0]:
                    #print "wn.params <- rp"
                    wn.params = rp
                else:
                    # contraction
                    if rv <= wv:
                        # inner contraction
                        np = (wp + cp)/2
                    else:
                        # outer contraction
                        np = (rp + cp)/2
                    #print "wn.params <- np"
                    wn.params = np
                    nv = wn.loss(by, x=bx)
                    if nv > wv:
                        wn.params = wp
                        for l, n in losses[1:]:
                            #print "contraction"
                            n.params = (n.params + bp)/2
                    else:
                        pass    # wn.params = np already done
    
            
                        
                    
        
if __name__ == '__main__':
    import math
    from mynnet import InputLayer, Linear, Tanh, L2Regression, Concat, Flatten, Conv, Reshape
    
    i1 = InputLayer((10,10,2))
    i2 = InputLayer((5,6,3))
    l1 = Flatten(Tanh(Conv(i1, 3,3,3)))
    l2 = Flatten(Tanh(Conv(i2, 2,2,3)))
    c = Concat((l1, l2))
    l3 = Tanh(Linear(c, 6))
    l4 = Reshape(Reshape(l3, (2,3)), (6,))
    
    m = Model((i1, i2), l4)
    
    x1 = np.random.random((1,)+i1.Shape)
    x2 = np.random.random((1,)+i2.Shape)
    y_ = np.random.random((1,)+l4.out_shape())
    
    eta = 0.03
    
    print y_
    y1 = m((x1, x2))
    print y1, m.loss((x1, x2), y_)
    
    loss = 1.0
    steps = 100
    while steps > 0 and loss > 1.e-3:
        m.train((x1, x2), y_, eta)
        y2 = m((x1, x2))
        loss = m.loss(y_)
        print y2, loss, math.log(loss)
        steps -= 1
