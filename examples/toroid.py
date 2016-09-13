import numpy as np
import random, math
from mynnet import Tanh, Linear, Model, InputLayer, L2Regression, Distort, AdaDeltaApplier, MomentumApplier, Trainer
from mynnet import TrainerCallbackDelegate

class ToroidGenerator:
    
    def generate(self, mb_size):
        out_x = []
        out_y = []
        for i in range(mb_size):
            kind = random.randint(0,1)
            if kind == 0:
                f = random.random()*math.pi*2
                z = random.random()*0.4 - 0.2
                x = 0.5*math.cos(f)
                y = 0.5*math.sin(f)
                out_x.append(np.array([x,y,z]))
                out_y.append(np.array([1.0]))
            else:
                f = random.random()*math.pi*2
                y = random.random()*0.4 - 0.2
                x = 0.5 + 0.5*math.cos(f)
                z = math.sin(f)
                out_x.append(np.array([x,y,z]))
                out_y.append(np.array([-1.0]))
        return np.array(out_x), np.array(out_y)

class BatchGenerator:
    
    def __init__(self):
        self.TG = ToroidGenerator()
        
    def batch(self, bsize):
        return self.TG.generate(bsize)
        
    def batches(self, bsize, randomize = False, max_samples = None):
        samples = 0
        while True:
            n = bsize
            if max_samples != None:
                n = min(n, max_samples - samples)
            if not n:   break
            out = self.batch(n)
            samples += n
            yield out

class CallbackDelegate(TrainerCallbackDelegate):
    
    def endOfEpochCallback(self, trainer, model, nsamples, epoch, nepochsamples, x, y, y_, loss, error):
        lst = [tuple(x) for x in model["Distort1"].W]
        lst.sort()
        print "W:"
        for t in lst:
            print t
        

def network():

    inp = InputLayer((3,))
    l1 = Tanh(Linear(inp, 4, applier=MomentumApplier(0.5)))
    l2 = Tanh(Linear(l1, 2, applier=MomentumApplier(0.5)))
    l3 = Tanh(Linear(l2, 1, applier=MomentumApplier(0.5)))
    nn = Model((inp,), l3, L2Regression(l3))
    
    return nn
    
def network_():

    inp = InputLayer((3,))
    d1 = Tanh(Distort(inp, 4, applier=MomentumApplier(0.5), name="Distort1"))
    d2 = Tanh(Distort(d1, 2, applier=MomentumApplier(0.5), name="Distort2"))
    o = Tanh(Distort(d2, 1, applier=MomentumApplier(0.5), name="Distort3"))
    nn = Model((inp,), o, L2Regression(o))
    
    return nn
    
nn = network()

Epochs = 100000
MBSize = 50
Eta = 0.2
Eta_decay = 1.0-1.0/Epochs

bg = BatchGenerator()
vx, vy_ = bg.batch(1000)

tr = Trainer(bg, (vx, vy_), (vx, vy_), nn, 8888)        #, callback_delegate=CallbackDelegate())

tr.startTraining(Epochs, Eta, eta_decay = Eta_decay, report_samples = 1000,
    train_mb_size = 100, epoch_limit = 10000, normalize_grads = True)

tr.wait()        
    
    