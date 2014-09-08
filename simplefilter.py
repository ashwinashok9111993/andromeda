__author__ = 'ashwin'

import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import scipy.misc as scm
import matplotlib.pyplot as p

mod = SourceModule("""
__global__ void rgb2gray(float *C,float *dest,float *r_img)
{
int a,b;
int M=C[0];
int N=C[1];
a =  threadIdx.x + blockIdx.x * blockDim.x;
b =  threadIdx.y + blockIdx.y * blockDim.y;

while(a<M){

while(b<N){

dest[a*N+b] = (r_img[(a)*N+b])+(r_img[(a)*N+b]);

b += blockDim.y*gridDim.y;
}
a += blockDim.x*gridDim.x;
}
}
""")

a = scm.imread('lenaG.jpg').astype(np.float32)
(M,N)=a.shape
print M,N
r_img = a.reshape(M*N, order='F')
dest=np.zeros_like(r_img)

rgb2gray = mod.get_function("rgb2gray")
rgb2gray(drv.In(np.float32([M,N])),drv.Out(dest), drv.In(r_img),block=(32, 32, 1), grid=(M/32,N/32,1))

dest=np.reshape(dest,(M,N), order='F')

p.gray()
p.imshow(dest)
p.show()
