import numpy as np
import scipy as sp
import seaborn as sns
import cupy as cp
import time

b = [np.array([[1.],[1.]]) for i in range(1_000_000)]
Λ  = [np.array([[1., 2.],[3., 1.]]) for i in range(1_000_000)]
c  = [1 for i in range(1_000_000)]
d  = [1.345 for i in range(1_000_000)]

s = time.time()
ζ = cp.random.gamma(cp.array(c)/2+1, scale = 1/(cp.array(d)/2))
σ2 = 1/ζ
mean = cp.array(b)
cov = cp.multiply(cp.repeat(ζ,2).reshape(-1,1),cp.asarray(sp.linalg.block_diag(*Λ)))
β = cp.random.multivariate_normal(mean.flatten(), cov).reshape(-1,1)
e = time.time()
print(e - s)