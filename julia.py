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

mod = SourceModule(
"""
__global__ void julia(float *pix)
#include<math.h>
#define INDEX(a, b) a*256+b
{
  int i = threadIdx.x+(blockIdx.x*(blockDim.x));
  int j = threadIdx.y+(blockIdx.y*(blockDim.y));

  float newRe, newIm, oldRe, oldIm;
  float  cRe = -0.7;
  float  cIm = 0.270;
  int maxIterations = 300;

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
            if((newRe * newRe + newIm * newIm) > 4) break;
        }
    pix[INDEX(i,j)] = id;

  i += blockDim.x * gridDim.x;
  j += blockDim.y * gridDim.y;

}
""")

julia = mod.get_function("julia")

M=256
N=256
pix = np.zeros(M*N)

