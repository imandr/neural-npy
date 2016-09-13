import numpy as np
from mynnet import Conv, Pool, Linear, LogRegression, L2Regression, Model, InputLayer, Reshape, Flatten, ParamMixin, Transpose
from mynnet import Tanh, Sigmoid, one_hot, ReLU, LSTM, LSTM_X, Concat, Conv, Pool_

def pool(nout):

	inp = InputLayer((9,8,5))
	pool = Pool(inp, (3,4))
	return Model(inp, pool, L2Regression(pool))

def    cnn(nout):
    inp = InputLayer((10,10))
    r = Reshape(inp, (10,10,1))
    c = Conv(r, 3, 3, 3)
    p = Pool_(c, (2,2))
    
    f = Flatten(p)
    loss = L2Regression(f)
    nn = Model(inp, f, loss)
    #nparams = 0
    #for l in nn.layers:
    #    if isinstance(l, ParamMixin):
    #        for p in l.params():
    #            nparams += p.size
    #print nparams
    return nn
    

nn = cnn(10)

#x = np.random.random((1,28,28))
#y_ = nn(x)
#x = np.random.random((1,28,28))

nn.check_gradients()

