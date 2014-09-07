__author__ = 'ashwin'

import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import numpy as np
from tictoc import *
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import pycuda.curandom
import pycuda.cumath
import matplotlib.pyplot as p

mod = SourceModule(
"""
__global__ void julia(float *pix)

#define INDEX(a, b) a*256+b
{
  int i = threadIdx.x+(blockIdx.x*(blockDim.x));
  int j = threadIdx.y+(blockIdx.y*(blockDim.y));

  float newRe, newIm, oldRe, oldIm;
  float  cRe = -0.01;
  float  cIm = 0.7;
  int maxIterations = 200;
  newRe =   0.05*(i)/12-0.05*12;
  newIm =   0.05*(j)/12;
  int id;
        //start the iteration process
        for(id = 0; id < maxIterations; id++)
        {
            //remember value of previous iteration
            oldRe = newRe;
            oldIm = newIm;
            //the actual iteration, the real and imaginary part are calculated
            newRe = oldRe * oldRe - oldIm * oldIm + cRe;
            newIm = 2 * oldRe * oldIm + cIm;
            //if the point is outside the circle with radius 2: stop
            if((newRe * newRe + newIm * newIm) > 3)
            {
            break;
            }
        }
        pix[INDEX(i,j)] = id;



}
""")

julia = mod.get_function("julia")

M=256
N=256
pix = np.zeros(M*N,order='F').astype(np.float32)
julia(drv.InOut(pix),block=(32,32, 1), grid=(8,8, 1))
pix=np.reshape(pix,(M,N), order='F').astype(np.float32)
print pix
p.imshow(pix)
p.show()