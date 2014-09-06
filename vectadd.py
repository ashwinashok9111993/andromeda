__author__ = 'ashwin'


import numpy as np
import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule
from tictoc import *

mod = SourceModule("""
__global__ void add_them(float *dest, float *a, float *b)
{
  const int i = threadIdx.x+(blockIdx.x*(blockDim.x));
  while(i < N)
  dest[i] = a[i] + b[i];
  tid += blockDim.x * gridDim.x;

}
""")

multiply_them = mod.get_function("add_them")

a = np.random.randn(65536).astype(np.float32)
b = np.random.randn(65536).astype(np.float32)

dest = np.zeros_like(a)

print 'gpu'
tic()
multiply_them(
        drv.Out(dest), drv.In(a), drv.In(b),
        block=(1024,1,1),grid=(64,1,1))
toc()

print 'cpu'
tic()
a+b
toc()


