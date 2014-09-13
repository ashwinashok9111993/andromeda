__author__ = 'ashwin'

import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import scipy.misc as scm
import matplotlib.pyplot as p

mod = SourceModule("""
__global__ void filter(float *C,float *dest,float *r_img)
{
int a,b;
int M=C[0];
int N=C[1];
a =  threadIdx.x + blockIdx.x * blockDim.x;


while(a<M*N){


if(a > 4 && a<(M*N-4)){
dest[a] = r_img[a]+r_img[a+1]+r_img[a+2]+r_img[a+3]+r_img[a+4]+r_img[a-1]+r_img[a-2]+r_img[a-3]+r_img[a-4];
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

filter = mod.get_function("filter")
filter(drv.In(np.float32([M,N])),drv.InOut(dest), drv.In(r_img),block=(1024, 1, 1), grid=(M*N/1024,1,1))

dest=np.reshape(dest,(M,N), order='F')
p.gray()
p.subplot(1,2,1)
p.imshow(a)
p.subplot(1,2,2)
p.imshow(dest)
p.show()
