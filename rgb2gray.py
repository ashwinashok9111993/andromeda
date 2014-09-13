__author__ = 'ashwin'

import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import scipy.misc as scm
import matplotlib.pyplot as p

mod = SourceModule("""
__global__ void rgb2gray(float *C,float *dest,float *r_img, float *g_img, float *b_img)
{
int a,b;
int M=C[0];
int N=C[1];
a =  threadIdx.x + blockIdx.x * blockDim.x;
b =  threadIdx.y + blockIdx.y * blockDim.y;
while(a<M){

while(b<N){

dest[a*N+b] = (0.299*r_img[a*N+b]+0.587*g_img[a*N+b]+0.114*b_img[a*N+b]);
__syncthreads();
b += blockDim.y*gridDim.y;
}
a += blockDim.x*gridDim.x;
}
}
""")

a = scm.imread('dice.png').astype(np.float32)
(M,N,L)=a.shape
print M,N

r_img = a[:, :, 0].reshape(M*N, order='F')
g_img = a[:, :, 1].reshape(M*N, order='F')
b_img = a[:, :, 2].reshape(M*N, order='F')
dest=np.zeros_like(r_img)
print dest

rgb2gray = mod.get_function("rgb2gray")
rgb2gray(drv.In(np.float32([M,N])),drv.Out(dest), drv.In(r_img), drv.In(g_img),drv.In(b_img),block=(32, 32, 1), grid=(M/32,N/32,1))

dest=np.reshape(dest,(M,N), order='F')

p.gray()
p.imshow(dest)
p.show()
