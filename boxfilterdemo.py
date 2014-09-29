__author__ = 'ashwin'

import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from pycuda.compiler import SourceModule
import scipy.misc as scm
import matplotlib.pyplot as p
from tictoc import *


#realrow = np.random.random([20,20]).astype(np.float32)
#print realrow.shape

realrow = scm.imread('lenaG.jpg').astype(np.float32)

(M,N)=realrow.shape
print realrow.shape



mod_copy_texture=SourceModule(
"""
texture<float,2>tex;
__global__ void  copy_texture_kernel(float *C,float * data)
 {
  int i = threadIdx.x+(blockIdx.x*(blockDim.x));
  int j = threadIdx.y+(blockIdx.y*(blockDim.y));
  int sum = 0;
  int M=C[0];
  int N=C[1];
  float A=C[2];
  int m,n=0;
   while(i<M)
  {
  while(j<N)
  {


  for(m=0;m<A;m++)
  {
  for(n=0;n<A;n++)
  {
  sum+=tex2D(tex,j-n,i-m);
  }
  }
  data[i*N+j] = sum/A;

  j += blockDim.y * gridDim.y;
  }
  i += blockDim.x * gridDim.x;
  }
}
""")

########
#get the kernel
########
copy_texture_func = mod_copy_texture.get_function("copy_texture_kernel")

#########
#Map the Kernel to texture object
#########
texref = mod_copy_texture.get_texref("tex")
cuda.matrix_to_texref(realrow , texref , order = "C")

#texref.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)
#texref.set_filter_mode()

A=5
gpu_output = np.zeros_like(realrow)
tic()
copy_texture_func(cuda.In(np.float32([M,N,A])),cuda.Out(gpu_output),block=(32,32, 1), grid=(M/32,N/32,1), texrefs=[texref])
print "time        ",toc()

print "Output"
print gpu_output

p.gray()
p.subplot(1,2,1)
p.imshow(realrow)
p.subplot(1,2,2)
p.imshow(gpu_output)
p.show()

