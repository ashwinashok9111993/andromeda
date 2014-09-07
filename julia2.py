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

mod = SourceModule("""

__global__ void julia(float *pix)
{

  int i = threadIdx.x+(blockIdx.x*(blockDim.x));
  int j = threadIdx.y+(blockIdx.y*(blockDim.y));

  float newRe, newIm, oldRe, oldIm;
  float  cRe = 0.01;
  float  cIm = 0.7;
  int maxIterations = 200;
  newRe =   0.5*(i)/2560;
  newIm =   0.5*(j)/2560;
  int id;
  while(i<2560)
  {
  while(j<2560)
  {
   for(id = 0; id < maxIterations; id++)
        {
            //remember value of previous iteration
            oldRe = newRe;
            oldIm = newIm;
            //the actual iteration, the real and imaginary part are calculated
            newRe = oldRe * oldRe - oldIm * oldIm + cRe;
            newIm = 2 * oldRe * oldIm + cIm;
            //if the point is outside the circle with radius 2: stop
            if((newRe * newRe + newIm * newIm) > 9)
            {
            break;
            }
        }
        pix[i*2560+j] = (id);

  j += blockDim.y * gridDim.y;
  }
  i += blockDim.x * gridDim.x;
  }


}
""")

julia = mod.get_function("julia")

M=2560
N=2560
pix = np.zeros(M*N,order='F').astype(np.float32)
julia(drv.InOut(pix),block=(32,32, 1), grid=(M/32,N/32, 1))
pix=np.reshape(pix,(M,N), order='F').astype(np.float32)
print pix
p.imshow(pix)
p.show()