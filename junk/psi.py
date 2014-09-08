__author__ = 'ashwin'
# -*- coding: utf-8 -*-
"""
Demonstrates GLVolumeItem for displaying volumetric data.

"""

## Add path to library (just for examples; you do not need this)

import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import numpy as np
from tictoc import *
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import pycuda.curandom
import pycuda.cumath
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import numpy as np

app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.opts['distance'] = 200
w.show()
w.setWindowTitle('pyqtgraph example: GLVolumeItem')

#b = gl.GLBoxItem()
#w.addItem(b)
g = gl.GLGridItem()
g.scale(10, 10, 1)
w.addItem(g)



mod = SourceModule(
"""

#include<math.h>
#define pi 3.14

__global__ void add_them(float *data){



  int i = threadIdx.x+(blockIdx.x*(blockDim.x));
   float x,y,z=0;
   int WIDTH = 100;
   int HEIGHT = 100;
   int th=0;
   int r,a0 =2;


while(i<100*100*100){


     z = (i/(WIDTH*HEIGHT));
     y = ((i%(WIDTH * HEIGHT))/WIDTH);
     x = i-(y*WIDTH) -(z*WIDTH*HEIGHT);


     if(sqrtf((x)*(x) + y*y + z*z) > 100)
     {data[i] = -10;}
     else
     {data[i] = 10;}
    //   data[i] = y;


  i += blockDim.x * gridDim.x;}

}
""")


add_them = mod.get_function("add_them")
data = np.zeros(100*100*100,order = 'F').astype(np.float32)
print data.shape
add_them(
        drv.InOut(data),
        block=(1024,1,1),grid=(64,1,1))
data=np.reshape(data,(100,100,100), order='F').astype(np.float32)



#data = np.fromfunction(psi, (100,100,200))
print data

positive = np.log(np.clip(data, 0, data.max())**2)
negative = np.log(np.clip(-data, 0, -data.min())**2)

d2 = np.empty(data.shape + (4,), dtype=np.ubyte)
d2[..., 0] = positive * (255./positive.max())
d2[..., 1] = negative * (255./negative.max())
d2[..., 2] = d2[...,1]
d2[..., 3] = d2[..., 0]*0.3 + d2[..., 1]*0.3
d2[..., 3] = (d2[..., 3].astype(float) / 255.) **2 * 255

d2[:, 0, 0] = [255,0,0,100]
d2[0, :, 0] = [0,255,0,100]
d2[0, 0, :] = [0,0,255,100]

v = gl.GLVolumeItem(d2)
#v.translate(-50,-50,-100)
w.addItem(v)

ax = gl.GLAxisItem()
w.addItem(ax)


## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
