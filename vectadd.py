__author__ = 'ashwin'


import numpy as np
import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule
from tictoc import *
import matplotlib.pyplot as p


mod = SourceModule(
"""
__global__ void add_them(float *dest, float *a, float *b)

#include<math.h>
{
  int i = threadIdx.x+(blockIdx.x*(blockDim.x));

  dest[i] = sin(a[i]) + sin(b[i]);
  i += blockDim.x * gridDim.x;

}
""")

multiply_them = mod.get_function("add_them")

###############################################################################
#
# We shall perform the benchmarking about 26 times, each time doubling the
# number of elements considered, from 2 to 2^26
#
###############################################################################


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
        block=(1024,1,1),grid=(64,1,1))

    gpu_time[i] = toc()

   # print 'cpu'
    tic()
    np.sin(a)+np.sin(b)
    cpu_time[i] = toc()


speedup = cpu_time/gpu_time

print 'maximum size of array ', N
print 'number of iteration ', len(speedup)

p.subplot(2, 1, 1)
p.plot((np.arange(0, 26)+1), cpu_time, label='cpu')
p.plot((np.arange(0, 26)+1), gpu_time,label='gpu')
p.ylabel("time in sec")
p.xlabel("array size in log(N)/log(2)")
p.legend()
p.subplot(2, 1, 2)
p.plot((np.arange(0, 26)+1), speedup, color = 'red')
p.ylabel("speed up")
p.xlabel("array size in log(N)/log(2)")
p.show()
