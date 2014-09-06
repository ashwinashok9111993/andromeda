__author__ = 'ashwin'

import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import numpy as np
from pycuda.curandom import rand
from tictoc import *
import pycuda.gpuarray as gpuarray
from pycuda.elementwise import ElementwiseKernel

##########################################################################
# In Monte-carlo method, we use a lot random number calculation and
# element wise operations. Those are best suited for the GPU
# therefore we take advantage of PyCuda's Numpy inspired GPUarray class
# This example also illustrates curand and Elementwise kernels
# this example is no way optimised!
# This example exists only to show that numpy can be adapted(-ish) to GPU
##########################################################################

circle = ElementwiseKernel(
        " float *x,float *y,float *z",
        " z[i]=floor(x[i]*x[i]+y[i]*y[i]);",
        " complex5",
        keep=True)
tic()
N = 80009999
print "Number of iterations considered",N
x_gpu = rand([N], dtype=np.float32, stream=None)
y_gpu = rand([N], dtype=np.float32, stream=None)
z_gpu = gpuarray.zeros_like(x_gpu)

circle(x_gpu,y_gpu,z_gpu)

print "gpu_pi   "  ,((N-gpuarray.sum(z_gpu))/N)*4
print "time     " ,toc()

##########################################################
# CPU implementation of monte-carlo.(adapted from Wikipedia)
##########################################################
tic()
x = np.random.rand(N)
y = np.random.rand(N)
z=np.floor((x)**2+(y)**2)
pi = ((N-np.sum(z))/N)*4
print "cpu_pi   ",pi
print "time     ",toc()




