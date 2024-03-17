#This is a Test program for solving linear equations

import numpy as np
import scipy as scp
from numpy import linalg

#initializing the Material Tensors of phase A
a = np.array([[1,2,3,4,5,6],[3,5,4,2,1,-1],[1,2,1,1,2,1],[3,1,2,-1,-2,1],[-1,6,2,3,-4,5],[1,6,2,5,3,4]])
b = np.array([1,2,3,4,5,6])
x = np.zeros((6,1),dtype=float) 

print(a)
print(b)




x = np.linalg.solve(a, b)
print(x)






