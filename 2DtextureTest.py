__author__ = 'ashwin'

import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from pycuda.compiler import SourceModule
import scipy.misc as scm
import matplotlib.pyplot as p

realrow = np.random.random([20,20]).astype(np.float32)

print realrow.shape

mod_copy_texture=SourceModule(
"""
texture<float,2>tex;
__global__ void  copy_texture_kernel(float * data) {
  int tx = threadIdx.x;
  int ty= threadIdx.y;
  data[tx*(20)+ty] = tex2D(tex,tx,ty);
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
copy_texture_func(cuda.Out(gpu_output),block=(20,20,1), texrefs=[texref])


print "Output"
print gpu_output

