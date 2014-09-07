__author__ = 'ashwin'

import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import numpy as np
from tictoc import *
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import pycuda.curandom
import pycuda.cumath

print pycuda.curandom.get_curand_version()
a =  pycuda.curandom.seed_getter_uniform(9)
b = pycuda.curandom.MRG32k3aRandomNumberGenerator(seed_getter=None)

tic()
N = 9999999
x = b.gen_uniform(N, dtype=np.float32 , stream=None)
y = b.gen_uniform(N, dtype=np.float32 , stream=None)

z = pycuda.cumath.floor((x.__pow__(2))+(y.__pow__(2)))
pi = ((N-gpuarray.sum(z))/N)*4

print "gpu_pi   ",pi
print "time     ",toc()


tic()
x =np.random.rand(N)
y =np.random.rand(N)
z =np.floor((x)**2+(y)**2)
pi=((N-np.sum(z))/N)*4
print "cpu_pi   ",pi
print "time     ",toc()