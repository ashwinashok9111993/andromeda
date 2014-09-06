__author__ = 'ashwin'

import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import numpy as np
from pycuda.curandom import rand
from tictoc import *
import pycuda.gpuarray as gpuarray
from pycuda.elementwise import ElementwiseKernel



circle = ElementwiseKernel(
        " float *x,float *y,float *z",
        " z[i]=floor(x[i]*x[i]+y[i]*y[i]);",
        "complex5",
        keep=True)
tic()
N = 80000012
print "Number of iterations considered",N
x_gpu = rand([N], dtype=np.float32, stream=None)
y_gpu = rand([N], dtype=np.float32, stream=None)
z_gpu = gpuarray.zeros_like(x_gpu)

circle(x_gpu,y_gpu,z_gpu)

print "gpu_pi   "  ,((N-gpuarray.sum(z_gpu))/N)*4
print "time     " ,toc()

tic()
x = np.random.rand(N)
y = np.random.rand(N)
z=np.floor((x)**2+(y)**2)
pi = ((N-np.sum(z))/N)*4
print "cpu_pi   ",pi
print "time     ",toc()




