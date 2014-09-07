__author__ = 'ashwin'

##########################################################################
# In Monte-carlo method,we take advantage of PyCuda's Numpy inspired
# GPUarray class
#
# this example is no way optimised!
# This example exists only to show that numpy can be adapted(-ish) to GPU
##########################################################################


import pycuda.driver as drv.__pow__
import pycuda.tools
import pycuda.autoinit
import numpy as np
from tictoc import *
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import pycuda.curandom
import pycuda.cumath

N = 9999999

print pycuda.curandom.get_curand_version()
a =  pycuda.curandom.seed_getter_uniform(9)
b = pycuda.curandom.XORWOWRandomNumberGenerator(seed_getter=None)

tic()
x = b.gen_uniform(N, dtype=np.float32 , stream=None)
y = b.gen_uniform(N, dtype=np.float32 , stream=None)
z = pycuda.cumath.floor((x**(2)).__add__(y**(2)))
pi = ((N-gpuarray.sum(z))/N)*4
print "gpuarray_pi  ",pi
print "time         ",toc()


tic()
x = gpuarray.GPUArray.get(b.gen_uniform(N, dtype=np.float32 , stream=None))
y = gpuarray.GPUArray.get(b.gen_uniform(N, dtype=np.float32 , stream=None))
z =np.floor((x)**2+(y)**2)
pi=((N-np.sum(z))/N)*4
print "cpu_pi      ",pi
print "time        ",toc()

