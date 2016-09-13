import cconv
import numpy as np

a = np.random.random((3, 4, 4, 2))
print a.shape, a
print "pool..."
a_pool, index = cconv.pool(a, 2, 2)
print a_pool.shape, a_pool
print index.shape, index
g = -a_pool*100
print "back..."
print g
x = cconv.pool_back(g, index, 2, 2, a.shape[1], a.shape[2])
print "done"

print "x:", x.shape, x.dtype, x



for _ in range(100000):
    a = np.random.random((10, 10, 40, 3))
    #print "pool..."
    a_pool, index = cconv.pool(a, 3, 3)
    g = -a_pool
    x = np.zeros_like(a)
    #print "back..."
    cconv.pool_back(g, index, 3, 3, a.shape[1], a.shape[2])

