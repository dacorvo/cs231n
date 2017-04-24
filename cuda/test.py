import numpy as np
from time import time

for n in (10, 100, 1000, 10000):
    A = np.random.random((n,n))
    B = np.random.random((n,n))
    t0 = time()
    C = A.dot(B)
    t1 = time()
    print("%dx%d Matrix multiplications done in %f" % (n,n,t1 - t0))
