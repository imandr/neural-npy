import numpy as np
import random


def one_hot(labels):
    classes = np.unique(labels)
    n_classes = classes.size
    one_hot_labels = np.zeros(labels.shape + (n_classes,))
    for c in classes:
        one_hot_labels[labels == c, c] = 1
    return one_hot_labels


def unhot(one_hot_labels):
    return np.argmax(one_hot_labels, axis=-1)


def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))


def sigmoid_d(x):
    s = sigmoid(x)
    return s*(1-s)


def tanh(x):
    return np.tanh(x)


def tanh_d(x):
    e = np.exp(2*x)
    return (e-1)/(e+1)

def relu(x):
    return np.maximum(0.0, x)


def relu_d(x):
    dx = np.zeros(x.shape)
    dx[x >= 0] = 1
    return dx

class BatchGenerator:
    
    def __init__(self, xdata, ydata):
        self.XData = xdata
        self.YData = ydata
        self.Size = len(self.XData)
        assert len(self.XData) == len(self.YData)
        self.I = 0
        
    def batch(self, bsize, increment=False, max_samples = None):
        imax = self.Size if max_samples is None else max_samples
        if self.I >= imax:  return None
        n = min(bsize, imax-self.I)
        x = np.array(self.XData[self.I:self.I + n])
        y = np.array(self.YData[self.I:self.I + n])
        if increment:   self.I += n
        return x, y
        
    def batches(self, bsize, randomize = False, max_samples = None):
        if randomize:
            lst = zip(self.XData, self.YData)
            random.shuffle(lst)
            self.XData, self.YData = zip(*lst)
        self.I = 0
        while True:
            out = self.batch(bsize, increment=True, max_samples = max_samples)
            if not out: break
            yield out

