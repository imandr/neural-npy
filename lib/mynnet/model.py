import numpy as np
import random, math, time
from mynnet import L2Regression, SaveModelMixin

class ParamSet(object):
    def __init__(self, lst):
        self.Params = lst       # list of tuples: [(W,b), (W,b), (W,V,b), ...]
        
    @property
    def params(self):
        return self.Params
        
    def copy(self):
        lst = []
        for tup in self.params:
            lst.append(tuple([p.copy() for p in tup]))
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
        

class Model(SaveModelMixin, object):
    
    def __init__(self, inputs, out, loss=None, rng=None, applier_class=None, applier_params=()):
        #print "NeuralNetwork.__init__()"
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        self.Inputs = inputs
        self.Out = out
        self.Loss = loss or L2Regression(self.Out)
        if rng is None:
            rng = np.random.RandomState()
        self.Loss.init(rng)
        
        if not applier_class is None:
            for l in self.layers:
                if l.params:
                    a = applier_class(*applier_params)
                    l.setApplier(a)
        
    def resetState(self):
        #print "model.resetState()"
        self.Loss.reset_state_rec()

    def compute(self, x, reset_state = True, step_id = None):
        if not isinstance(x, (list, tuple)):
            x = [x]
        if reset_state: self.resetState()
        if step_id is None:
            step_id = time.time()
        self.X = x
        for ii, xx in zip(self.Inputs, x):
            ii.set(xx, step_id)
        self.Y = self.Loss.compute(step_id)
        self.StepID = step_id
        return self.Y
        
    __call__ = compute
            
    def train(self, x, y_, eta, normalize_grads = False, recompute = True, reset_state = True):
        # x is list of inputs

        if not isinstance(x, (list, tuple)):
            x = [x]
        
        ninputs = len(x)
        assert ninputs == len(self.Inputs), "Invalid number of inputs: %d, expected %d" % (ninputs, len(self.Inputs))
        mb_size = len(x[0])
        
        #print "mode.train"
        if recompute:   
            y = self.compute(x, reset_state = reset_state)
        else:           
            y = self.Y
            
        #print "train: x:", x[0].shape, "    y:",y.shape, "   y_:",y_.shape
            
        step_id = self.StepID
        #print "train: y_, y:", y_.shape, y.shape
        loss = self.Loss.loss(y_, y)
        error = self.Loss.error(y_, y)
        
        #print "loss,error=", loss, error
        
        grad = self.Loss.input_grad(y_, y)
        self.GY = grad
        #print "train: gy:", self.GY.shape
        #print "Model.train(): GY=", self.GY
        
        #print "train: grad=", grad.shape
        self.Out.reset_backprop_rec()
        #print "model.train: grad=", grad
        self.Out.backprop(grad, step_id, normalize_grads = normalize_grads)
        self.GX = [inp.gX[0] for inp in self.Inputs]        # our loss is always normalized per sample in the minibatch
                                                                    # so individual x's gradient needs to be divided by mb size
        #print "model.train: GX=", gx
        if eta:
            self.Out.apply_backprop_rec(eta, step_id)
        return y, loss, error
        
    def loss(self, y_, x=None):
        if not x is None:
            self.compute(x)
        return self.Loss.loss(y_, self.Y)
        
    average_loss = loss     # average over the minibatch
        
    def total_loss(self, y_, x=None):
        if not x is None:
            self.compute(x)
        return self.Loss.total_loss(y_, self.Y)
        
    def display_loss(self, y_, x=None):
        if not x is None:
            self.compute(x)
        return self.Loss.display_loss(y_, self.Y)
        
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
                    new_scan += l.Inputs
            to_scan = new_scan
        return all_layers
        
    def __getitem__(self, name):
        # finds layer by name
        for l in self.layers:
            if l.Name == name:  return l
        raise KeyError, "Layer %s not found" % (name,)
        
    def __get_params(self):
        #print "__get_params"
        return ParamSet([l.params for l in self.layers if l.params])
        
    def __set_params(self, ps):
        #print "Model.set_params"
        lst = [l for l in self.layers if l.params]
        for l, p in zip(lst, ps.params):
            l.params = p
            
    params = property(__get_params, __set_params)

    def nparams(self):
        nparams = 0
        for l in self.layers:
            for p in l.params:
                nparams += p.size
        return nparams
    
        
    def checkGradients(self, X = None, Y_ = None, check_params = True, check_inputs = True, mb_size=3,
                    random_params = 20, random_inputs = 20):
        """ Helper function to test the parameter gradients for
        correctness. """
        
        
        if X == None:
            X = [np.random.random((mb_size,) + inp.shape) for inp in self.Inputs]
        if Y_ == None:
            xr = [np.random.random((mb_size,) + inp.shape) for inp in self.Inputs]
            Y_ = self.compute(xr)
            
        Y = self.compute(X)
        self.train(X, Y_, 0.0)       # force grad calculation, but do not actually update any weights
        
        #print "check_gradients: GX=", self.GX
        
        if check_params:
            print "checking parameters..."
            nbad = 0
            dp = 0.001
            for l in self.layers:
                if l.params:
                    nparams = sum([p.size for p in l.params])
                    print "Checking layer %s with %d parameters" % (l, nparams)
                    for p, (param, paramg) in enumerate(zip(l.params, l.param_grads())):
                        param_shape = param.shape
                        print "param shape:", param_shape
                        pflat = param.flat
                        #print pflat
                        pgflat = paramg.copy().flat
                        rng = range(param.size)
                        inxs = np.unravel_index(rng, param.shape)
                        inxs = zip(*inxs)
                        if not random_params is None and param.size > random_params:
                            rng = random.sample(rng, random_params)
                        for i in rng:
                            #print "y0: ", Y
                            v0 = pflat[i]
                            
                            pflat[i] = v0-dp
                            lminus = self.loss(Y_, X)       # loss is average over minibatch

                            pflat[i] = v0+dp
                            lplus = self.loss(Y_, X)

                            pflat[i] = v0
                    
                            grad_c = (lplus-lminus)/(dp*2)
                            grad_a = pgflat[i]
                            delta = grad_a-grad_c
                            if True or grad_c != grad_a and (abs(delta) > 1.e-4 or \
                                    grad_a == 0.0 and grad_c != 0.0 or \
                                    abs(grad_c/grad_a-1) > 1.0e-2):
                                nbad += 1
                                print "param %d, index %d (%s): grad_c:%s, grad_a:%s, delta:%s, grad_c/grad_a=%f" % (p, i, inxs[i], grad_c, grad_a, delta, (grad_c+1e-6)/(grad_a+1e-6))
            if not nbad:
                print "No errors found ------------------"
        if check_inputs:
            print
            print "Checking inputs ..."
            nbad = 0
        
            dx = 0.0001
        
            for j, (x, gx_a) in enumerate(zip(self.X, self.GX)):
                    #print "gx_a=", gx_a
                    rng = range(x.size) 
                    if not random_inputs is None and x.size > random_inputs:
                        rng = random.sample(rng, random_inputs)
                    for i in rng:
                
                        gradx_a = gx_a.flat[i]

                        x0 = x.flat[i]
            
                        x.flat[i] = x0 + dx
                        self.compute(X)
                        lplus = self.total_loss(Y_)       

                        x.flat[i] = x0 - dx
                        self.compute(X)
                        lminus = self.total_loss(Y_)
            
                        gradx_c = (lplus-lminus)/(dx*2)
                        x.flat[i] = x0
                        
                        delta = gradx_a-gradx_c
                        if True or gradx_a != gradx_c and (abs(delta) > 1.0e-4 or abs(gradx_a/gradx_c-1) > 1.0e-2):
                            print "x[%d][%d] gradx_c:%s, gradx_a:%s, delta:%s, %f" % (j,i, gradx_c, gradx_a, gradx_a-gradx_c, gradx_a/gradx_c)
                            nbad += 1

            if not nbad:
                print "No errors found ------------------"
            
        self.compute(X)
        
    @staticmethod
    def trainSimplex____(nets, batches):
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
    
    i1 = InputLayer((10,))
    l1 = Tanh(Linear(i1, 20))
    l2 = Tanh(Linear(l1, 5))

    m = Model((i1,), l2, L2Regression(l2))
    
    m.check_gradients()
    
    x1 = np.random.random((1,)+i1.shape)
    y_ = np.random.random((1,)+l2.shape)
    
    eta = 0.03
    
    print y_
    y1 = m((x1,))
    print y1, m.loss(y_)
    
    loss = 1.0
    steps = 100
    while steps > 0 and loss > 1.e-3:
        m.train((x1,), y_, eta)
        y2 = m((x1,))
        loss = m.loss(y_)
        print y2, loss, math.log(loss)
        steps -= 1
