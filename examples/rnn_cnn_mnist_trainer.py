#!/usr/bin/env python
# coding: utf-8

import time, random
import numpy as np
import sklearn.datasets
from mynnet import LSTM, Transpose, Concat, Linear, LogRegression, Model, ParamMixin, InputLayer, one_hot
from mynnet import Sigmoid, Tanh, AdaDeltaApplier, Reshape, ReLU, Pool, Conv, Flatten, Trainer, BatchGenerator
from mynnet import MomentumApplier, TrainerCallbackDelegate


def buildNetwork_cnn(nout):
    i = InputLayer((28, 28))
    ir = Reshape(i, (28,28,1), name="Reshaped input")
    c1 = ReLU(
            Pool(Conv(ir, 5, 5, 16, name = "Conv1"),
                (2, 2),
                mode='max', name="Pool1"
            )
        )
    c2 = ReLU(
            Pool(Conv(c1, 5, 5, 30, name = "Conv2"),
                (2, 2),
                mode='max', name="Pool2"
            )
        )
    f = Flatten(c2)
    l1 = Tanh(Linear(f, 76, name="Hidden linear"))
    l2 = Linear(l1, nout,  name="Top linear")
    nn = Model(i, l2, LogRegression(l2), applier_class=AdaDeltaApplier)
    return nn
    
def buildNetwork_cnn_20k(nout):
    i = InputLayer((28, 28))
    ir = Reshape(i, (28,28,1), name="Reshaped input")
    c1 = ReLU(
            Pool(Conv(ir, 5, 5, 8, name = "Conv1"),
                (2, 2),
                mode='max', name="Pool1"
            )
        )
    c2 = ReLU(
            Pool(Conv(c1, 5, 5, 15, name = "Conv2"),
                (2, 2),
                mode='max', name="Pool2"
            )
        )
    f = Flatten(c2)
    l1 = Tanh(Linear(f, 38, name="Hidden linear"))
    l2 = Linear(l1, nout,  name="Top linear")
    nn = Model(i, l2, LogRegression(l2), applier_class=AdaDeltaApplier)
    return nn
    


def buildNetwork_lstm_xy2_extra_hidden(nout):
    ix = InputLayer((28, 28))
    iy = Transpose(ix, (1,0))
    
    # x path
    
    rnn_width = 40
    
    rx = Tanh(
        LSTM(ix, rnn_width, rnn_width, name="LSTM X1")
    )
    rx2 = Tanh(
        LSTM(rx, rnn_width, rnn_width, last_row_only = True, name="LSTM X2")
    )
    
    ry = Tanh(
        LSTM(iy, rnn_width, rnn_width, name="LSTM Y1")
    )
    ry2 = Tanh(
        LSTM(ry, rnn_width, rnn_width, last_row_only = True, name="LSTM Y2")
    )
        
    r = Concat((rx2, ry2))
    h = Tanh(Linear(r, 20))
    out = Linear(h, nout)
    nn = Model(ix, out, LogRegression(out), applier_class=AdaDeltaApplier)
    return nn
    
def buildNetwork_lstm_xy3_extra_hidden(nout):
    ix = InputLayer((28, 28))
    iy = Transpose(ix, (1,0))
    
    # x path
    
    rnn_width = 32
    
    rx = Tanh(
        LSTM(ix, rnn_width, rnn_width, name="LSTM X1")
    )
    rx2 = Tanh(
        LSTM(rx, rnn_width, rnn_width, name="LSTM X2")
    )
    rx3 = Tanh(
        LSTM(rx2, rnn_width, rnn_width, last_row_only = True, name="LSTM X3")
    )
    
    ry = Tanh(
        LSTM(iy, rnn_width, rnn_width, name="LSTM Y1")
    )
    ry2 = Tanh(
        LSTM(ry, rnn_width, rnn_width, name="LSTM Y2")
    )
    ry3 = Tanh(
        LSTM(ry2, rnn_width, rnn_width, last_row_only = True, name="LSTM Y3")
    )
        
    r = Concat((rx3, ry3))
    h = Tanh(Linear(r, 20))
    out = Linear(h, nout)
    nn = Model(ix, out, LogRegression(out), applier_class=AdaDeltaApplier)
    return nn
    
def buildNetwork_lstm_xy2(nout):
    ix = InputLayer((28, 28))
    iy = Transpose(ix, (1,0))
    
    # x path
    
    rnn_width = 40
    
    rx = Tanh(
        LSTM(ix, rnn_width, rnn_width, name="LSTM X1")
    )
    rx2 = Tanh(
        LSTM(rx, rnn_width, rnn_width, last_row_only = True, name="LSTM X2")
    )
    
    ry = Tanh(
        LSTM(iy, rnn_width, rnn_width, name="LSTM Y1")
    )
    ry2 = Tanh(
        LSTM(ry, rnn_width, rnn_width, last_row_only = True, name="LSTM Y2")
    )
        
    r = Concat((rx2, ry2))
    out = Linear(r, nout)
    nn = Model(ix, out, LogRegression(out), applier_class=AdaDeltaApplier)
    return nn
    
def buildNetwork_lstm_xy_reduction(nout):
    ix = InputLayer((28, 28))
    iy = Transpose(ix, (1,0))
    
    # x path
    
    rnn_width = 65
    
    rx = Tanh(LSTM(ix, nout, rnn_width, last_row_only = True, name="LSTM X1"))
    
    ry = Tanh(LSTM(iy, nout, rnn_width, last_row_only = True, name="LSTM Y1"))
        
    c = Concat((rx, ry))
    out = Linear(c, nout)
    nn = Model(ix, out, LogRegression(out), applier_class=AdaDeltaApplier)
    return nn
    
def buildNetwork_lstm_xy(nout):
    ix = InputLayer((28, 28))
    iy = Transpose(ix, (1,0))
    
    # x path
    
    rnn_width = 65
    
    rx = Tanh(
        LSTM(ix, rnn_width, rnn_width, last_row_only = True, name="LSTM X1")
    )

    ry = Tanh(
        LSTM(iy, rnn_width, rnn_width, last_row_only = True, name="LSTM Y1")
    )

    r = Concat((rx, ry))
    #h = Tanh(Linear(r, 20))
    out = Linear(r, nout)
    nn = Model(ix, out, LogRegression(out), applier_class=AdaDeltaApplier)
    return nn
    
def buildNetwork_lstm_xyxryr(nout):
    ix = InputLayer((28, 28))
    iy = Transpose(ix, (1,0))
    
    # x path
    
    rnn_width = 42
    
    rx = Tanh(
        LSTM(ix, rnn_width, rnn_width, last_row_only = True, name="LSTM X")
    )

    rxr = Tanh(
        LSTM(ix, rnn_width, rnn_width, reversed=True, last_row_only = True, name="LSTM XR")
    )

    ry = Tanh(
        LSTM(iy, rnn_width, rnn_width, last_row_only = True, name="LSTM Y1")
    )

    ryr = Tanh(
        LSTM(iy, rnn_width, rnn_width, reversed=True, last_row_only = True, name="LSTM YR")
    )

    r = Concat((rx, ry, rxr, ryr))
    #h = Tanh(Linear(r, 20))
    out = Linear(r, nout)
    nn = Model(ix, out, LogRegression(out), applier_class=AdaDeltaApplier)
    return nn
    
def buildNetwork_lstm_xy_plus_hidden(nout):
    ix = InputLayer((28, 28))
    iy = Transpose(ix, (1,0))
    
    # x path
    
    rnn_width = 64
    
    rx = Tanh(
        LSTM(ix, rnn_width, rnn_width, last_row_only = True, name="LSTM X1")
    )

    ry = Tanh(
        LSTM(iy, rnn_width, rnn_width, last_row_only = True, name="LSTM Y1")
    )

    r = Concat((rx, ry))
    h = Tanh(Linear(r, 20))
    out = Linear(h, nout)
    nn = Model(ix, out, LogRegression(out), applier_class=AdaDeltaApplier)
    return nn
    
def buildNetwork_lstm_x2(nout):
    ix = InputLayer((28, 28))
    
    # x path
    
    rnn_width = 59
    
    rx = Tanh(
        LSTM(ix, rnn_width, rnn_width, name="LSTM X1")
    )
    rx2 = Tanh(
        LSTM(rx, rnn_width, rnn_width, last_row_only = True, name="LSTM X2")
    )
    
    #h = Tanh(Linear(rx2, 38, applier=MomentumApplier(0.2)))
    out = Linear(rx2, nout)
    nn = Model(ix, out, LogRegression(out), applier_class=AdaDeltaApplier)
    return nn
    
def buildNetwork_lstm_x(nout):
    ix = InputLayer((28, 28))
    
    # x path
    
    rnn_width = 97
    
    rx = Tanh(
        LSTM(ix, rnn_width, rnn_width, last_row_only = True, name="LSTM X")
    )
    
    #h = Tanh(Linear(rx2, 38, applier=MomentumApplier(0.2)))
    out = Linear(rx, nout)
    nn = Model(ix, out, LogRegression(out), applier_class=AdaDeltaApplier)
    return nn
    
def buildNetwork_lstm_x2_extra_hidden(nout):
    ix = InputLayer((28, 28))
    
    # x path
    
    rnn_width = 59
    
    rx = Tanh(
        LSTM(ix, rnn_width, rnn_width, name="LSTM X1")
    )
    rx2 = Tanh(
        LSTM(rx, rnn_width, rnn_width, last_row_only = True, name="LSTM X2")
    )
    
    h = Tanh(Linear(rx2, nout*2))
    out = Linear(h, nout)
    nn = Model(ix, out, LogRegression(out), applier_class=AdaDeltaApplier)
    return nn
    
    
def buildNetwork_lstm_linked(nout):
    ix = InputLayer((28, 28))
    iy = Transpose(ix, (1,0))
    
    inp = Concat((ix, iy), axis=1)
    print inp.InShapes, inp.out_shape()
    
    # x path
    
    rnn_width = 80
    
    ax = AdaDeltaApplier()
    
    r = Tanh(
        LSTM(inp, 60, rnn_width, last_row_only = True, name="LSTM XY", applier=ax)
    )
    h = Tanh(Linear(r, 18, applier=AdaDeltaApplier()))
    out = Sigmoid(Linear(h, nout, applier=AdaDeltaApplier()))
    nn = Model(ix, out, LogRegression(out))
    return nn

def printImage(img):
    ny, nx = img.shape
    print '+' + '-'*nx + '+'
    for row in img:
        s = '|'
        for x in row:
            v = ' '
            if x > 0.1: v = '.'
            if x > 0.5: v = '+'
            if x > 0.9: v = '#'
            s += v
        print s + '|'
    print '+' + '-'*nx + '+'
    

def shift(data, dist):
    data = data.reshape((-1, 28, 28))

    for i, img in enumerate(data):
        done = False
        while  not done:
            dx = random.randint(-dist, dist)
            dy = random.randint(-dist, dist)
            done = dx or dy
            
        #print img.shape
        
        #if i < 3:
        #    printImage(img)
        if dx != 0:
            #print "shift x:", dx
            img[:,:] = np.roll(img, dx, axis=1)

            #if i < 3:
            #    printImage(img)
            
            if dx > 0:
                img[:,:dx] = 0.0
            else:
                img[:,dx:] = 0.0
            
            #if i < 3:
            #    printImage(img)

        if dy != 0:
            #print "shift y:", dy
            img[:,:] = np.roll(img, dy, axis=0)

            #if i < 3:
            #    printImage(img)

            if dy > 0:
                img[:dy,:] = 0.0
            else:
                img[dy:,:] = 0.0
        
        #if i < 3:
        #    printImage(img)

class CallbackDelegate(TrainerCallbackDelegate):
    
    def __init__(self, threshold, seqlen):
        self.T = threshold
        self.S = seqlen
        self.N = 0
        self.T0 = time.time()
    
    def reportCallback(self, trainer, model, nsamples, epoch, nepochsamples, bx, by, y, tloss, terror):
        #print "train    samples/loss/error:", nsamples, tloss, terror
        pass

    def endOfEpochCallback(self, trainer, model, nsamples, epoch, nepochsamples, bx, by, y, tloss, terror):
        #print "train    samples/loss/error:", nsamples, tloss, terror
        print "End of epoch. %d samples" % (nsamples,)
        
    def validateCallback(self, trainer, model, nsamples, epoch, nepochsamples, vx, vy, y, vloss, verror):
        print "validate time/samples/loss/error:", time.time() - self.T0, nsamples, vloss, verror
        return
        if verror >= self.T:
            self.N = 0
        else:
            self.N += 1
            if self.N >= self.S:
                print "stopping"
                trainer.stopTraining()

def run():
    # Fetch data
    mnist = sklearn.datasets.fetch_mldata('MNIST original', data_home='./data')
    targets = one_hot(mnist.target)
    print "data loaded"
    split = 60000
    
    shift(mnist.data, 2)
    
    X_train = np.reshape(mnist.data[:split], (-1, 28, 28))/255.0
    y_train = targets[:split]
    X_test = np.reshape(mnist.data[split:], (-1, 28, 28))/255.0
    y_test = targets[split:]
    
    #y_test = [y for y in y_test]
    #random.shuffle(y_test)
    #y_test = np.array(y_test)
    
    n_classes = y_train.shape[1]
    
    bg_train = BatchGenerator(X_train, y_train)
    bg_test = (X_test, y_test)
    
    # Setup convolutional neural network
    nn = buildNetwork_cnn(n_classes)
    print "Network witn %d parameters created" % (nn.nparams(),)
    #nn.check_gradients()

    # Train neural network
    
    batch_size = 100
    epochs = 20
    eta = 1.0
    deta = 1-1.0/epochs
    n_train_samples = 40000
    
    nimages = 0
    cb = CallbackDelegate(0.1, 2)
    trainer = Trainer(bg_train, bg_test, bg_test, nn, 8888, callback_delegate=cb)
    
    trainer.startTraining(epochs, eta, 
                    eta_decay = eta/epochs,
                    report_samples = 1000,
                    train_mb_size = batch_size, validate_samples = 20000,
                    randomize = True, normalize_grads = False)

    trainer.wait()
    print "=============== training is complete ==============="
    while True:
        time.sleep(10)      # keep the web server running


if __name__ == '__main__':
    run()
