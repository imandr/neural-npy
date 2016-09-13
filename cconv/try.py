import scipy.signal as sig
import myconv, time
import numpy as np

def conv_npy(a, f, mode):
    na, ka, ya, xa = a.shape
    nf, kf, yf, xf = f.shape
    # assume "valid" mode for now
    lst = []
    if mode == 'valid':
        xz = xa - xf + 1
        yz = ya - yf + 1
    else:   
        # full
        xz = xa + xf - 1
        yz = ya + yf - 1
    #print yz, xz
    for aa in a:
        for ff in f:
            for pa in aa:
                for pf in ff:
                    c = sig.convolve2d(pa, pf[::-1, ::-1], mode)
                    #print c.shape
                    lst.append(c)
    out = np.array(lst).reshape((na, nf, ka, kf, yz, xz))
    return np.trace(out, axis1=2, axis2=3)    

def try_values():
    a=np.random.random((3,2,5,5))
    f=np.random.random((3,2,2,2))
    #f[0,0,1,1] = 1.0

    conv = conv_npy(a, f, "valid")

    print "conv by numpy:", conv.shape, conv


    c = myconv.convolve(a, f, 0)
    print "my conv:",c.shape, c
    
def try_time():
    a=np.random.random((3,100,100,3))
    f=np.random.random((5,5,3,10))

    at = a.transpose((0,3,1,2))
    ft = f.transpose((3,2,0,1))

    t0 = time.time()
    
    for i in range(10):
        myconv.convolve(at, ft, 1)
    
    print time.time() - t0

    t0 = time.time()
    
    for i in range(10):
        conv_npy(at, ft, "full")
    
    print time.time() - t0

if __name__ == '__main__':
    try_values()    
    