__author__ = 'ashwin'

import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from pycuda.compiler import SourceModule
import scipy.misc as scm
import matplotlib.pyplot as p


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
  int gx=0;
  int gy=0;
  int M=C[0];
  int N=C[1];
   while(i<M)
  {
  while(j<N)
  {
  gx = -tex2D(tex,j-1,i-1)-2*tex2D(tex,j-1,i)-tex2D(tex,j-1,i+1)+tex2D(tex,j+1,i-1)+tex2D(tex,j+1,i+1)+2*tex2D(tex,j+1,i);
  gy = -tex2D(tex,j-1,i-1)-tex2D(tex,j,i-1)-tex2D(tex,j+1,i-1)+tex2D(tex,j-1,i+1)+2*tex2D(tex,j,i+1)+tex2D(tex,j+1,i+1);
  data[i*N+j] = sqrtf(gx*gx+gy*gy);
  __syncthreads();
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


gpu_output = np.zeros_like(realrow)
copy_texture_func(cuda.In(np.float32([M,N])),cuda.Out(gpu_output),block=(32,32, 1), grid=(M/32,N/32,1), texrefs=[texref])


print "Output"
print gpu_output

p.gray()
p.subplot(1,2,1)
p.imshow(realrow)
p.subplot(1,2,2)
p.imshow(gpu_output)
p.show()

