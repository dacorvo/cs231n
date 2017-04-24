import numpy as np
import cudarray as ca
from time import time

n, p = int(2e3), int(40e3)
A = np.random.randn(n, p)
B = np.random.randn(p, n)
t0 = time()
np.dot(A,B)
t1 = time()
print("Numpy %f" % (t1-t0))

A_ca = ca.random.normal(size=(n, p))
B_ca = ca.random.normal(size=(p, n))
t0 = time()
ca.dot(A_ca, B_ca)
t1 = time()
print("CUDArray%f" % (t1-t0))
