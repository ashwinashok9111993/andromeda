__author__ = 'ashwin'

mod = SourceModule(
"""
__global__ void monte(float *q, float *r, float *s)


{
  int i = threadIdx.x+(blockIdx.x*(blockDim.x));
  q[i] =floor(r[i]*r[i]+s[i]*s[i]);
  i += blockDim.x * gridDim.x;

}
""")

tic()

r = gpuarray.GPUArray.get(x)
s = gpuarray.GPUArray.get(y)
q = np.zeros_like(r)


monte = mod.get_function("monte")
monte(
        drv.Out(q), drv.In(r), drv.In(s),
        block=(1024,1,1),grid=(64,1,1))

pi = ((N-np.sum(q))/N)*4
print pi," in ", toc()," seconds "





from pycuda.elementwise import ElementwiseKernel
lin_comb = ElementwiseKernel(
        "float *q, float *r, float *s",
        "q[i] =floor(r[i]*r[i]+s[i]*s[i])",
        "linear_combination")
t = gpuarray.zeros(N,dtype=float)
lin_comb(t,b.gen_uniform(N, dtype=np.float32 , stream=None),b.gen_uniform(N, dtype=np.float32 , stream=None))
pi = ((N-gpuarray.sum(t))/N)*4
