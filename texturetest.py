__author__ = 'ashwin'

import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from pycuda.compiler import SourceModule

realrow = np.array([1.0,2.0,3.0,4.0,5.0],dtype=np.float32).reshape(1,5)
print realrow.shape

mod_copy_texture=SourceModule(
"""
texture<float,1>tex;
__global__ void  copy_texture_kernel(float * data) {
  int ty = threadIdx.y;
  data[ty] = tex1D(tex,(float)(ty)/1.0f);
}
""")


copy_texture_func = mod_copy_texture.get_function("copy_texture_kernel")
texref = mod_copy_texture.get_texref("tex")

cuda.matrix_to_texref(realrow , texref , order = "C")

#texref.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)
#texref.set_filter_mode()

gpu_output = np.zeros_like(realrow)

copy_texture_func(cuda.Out(gpu_output),block=(1,5,1), texrefs=[texref])


print "Output"
print gpu_output

