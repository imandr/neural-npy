import numpy as np
from mynnet import Conv, Pool, Linear, LogRegression, L2Regression, Model, InputLayer, Reshape, Flatten, ParamMixin, Transpose
from mynnet import Tanh, Sigmoid, one_hot, ReLU, LSTM, LSTM_X, Concat, Conv

    
def cnn(nout):
    inp = InputLayer((10,10))
    r = Reshape(inp, (10,10,1))
    c = Conv(r, 5, 5, 3)
    #p = Tanh(Pool(c, (4,4)))
    
    f = Flatten(c)
    loss = L2Regression(f)
    nn = Model(inp, f, loss)
    #nparams = 0
    #for l in nn.layers:
    #    if isinstance(l, ParamMixin):
    #        for p in l.params():
    #            nparams += p.size
    #print nparams
    return nn
    
def rnn(nout):
    inp_x = InputLayer((10, 10))
    #inp_y = Transpose(inp_x, (1,0))
    lstm_x = Flatten(Sigmoid(LSTM_X(inp_x, nout, 30)))
    #lstm_y = Tanh(LSTM(inp_y, 30, 30, last_row_only = True))
    #c = Concat((lstm_x, lstm_y))
    #o = Sigmoid(Linear(c, nout))
    return Model(inp_x, lstm_x, L2Regression(lstm_x))
    
def rnn(nout):
    inp_x = InputLayer((10, 10))
    inp_y = Transpose(inp_x, (1,0))
    inp_c = Concat((inp_x, inp_y), axis=1)
    #inp_y = Transpose(inp_x, (1,0))
    lstm = Flatten(Sigmoid(LSTM(inp_c, nout, 30)))
    #lstm_y = Tanh(LSTM(inp_y, 30, 30, last_row_only = True))
    #c = Concat((lstm_x, lstm_y))
    #o = Sigmoid(Linear(c, nout))
    return Model(inp_x, lstm, L2Regression(lstm))
    

    
    
def buildNetwork_():
    
    i1 = InputLayer((3,))
    i2 = InputLayer((3,))
    c = Concat((i1, i2))
    #print c.out_shape()
    return Model([i1, i2], c, L2Regression(c))

nn = cnn(10)

#x = np.random.random((1,28,28))
#y_ = nn(x)
#x = np.random.random((1,28,28))

nn.check_gradients()

