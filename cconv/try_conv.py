import cconv
import numpy as np
a=np.random.random((1,2,3))
b=np.random.random((3,2,1))
c=np.random.random((3,2,1))
cconv.convolve(a,b,c)
