__author__ = 'ashwin'


import numpy as np
import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule
from tictoc import *
import matplotlib.pyplot as p

mod = SourceModule("""
__global__ void add_them(float *dest, float *a, float *b)

#include<math.h>
{
  int i = threadIdx.x+(blockIdx.x*(blockDim.x));

  dest[i] = (a[i]) + (b[i]);
  i += blockDim.x * gridDim.x;

}
""")

multiply_them = mod.get_function("add_them")


speedup  = np.zeros(26)
gpu_time = np.zeros(26)
cpu_time = np.zeros(26)
for i in xrange(0,26):
    N = 2**(i+1)
   # print 'N = ' +str(N)
    a = np.random.rand(N).astype(np.float32)
    b = np.random.rand(N).astype(np.float32)

    dest = np.zeros_like(a)

   # print 'gpu'
    tic()
    multiply_them(
        drv.Out(dest), drv.In(a), drv.In(b),
        block=(128,1,1),grid=(128,1,1))

    gpu_time[i] = toc()

   # print 'cpu'
    tic()
    np.sin(a)+np.sin(b)
    cpu_time[i] = toc()


speedup = cpu_time/gpu_time

print N,len(speedup)
print "speed up is " + str(speedup)

p.subplot(2,1,1)
p.plot(np.arange(0,26)+1,cpu_time,np.arange(0,26)+1,gpu_time)
p.subplot(2,1,2)
p.plot(np.arange(0,26)+1,speedup)
p.show()
