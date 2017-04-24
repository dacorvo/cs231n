import numpy as np
import cudamat as cm
from time import time

n, p = int(2e3), int(40e3)
A = np.random.randn(n, p)
B = np.random.randn(p, n)
t0 = time()
np.dot(A,B)
t1 = time()
print("Numpy: %f" % (t1-t0))

cm.cublas_init()
cm.CUDAMatrix.init_random()
A_cm = cm.empty((n, p)).fill_with_randn()
B_cm = cm.empty((p, n)).fill_with_randn()
t0 = time()
A_cm.dot(B_cm)
t1 = time()
print("Cudamat: %f" % (t1-t0))
#cm.cublas_shutdown()
