#This is a Test program for constrained minimization

import numpy as np
import scipy as scp
from scipy.optimize import minimize
import math

#initializing the Material Tensors of phase A
F0 = np.zeros((3,2),dtype=float) 
Q = np.zeros((3,3),dtype=float) 

#Inputting the variables
F0[0,0] = -1
F0[1,0] = 3
F0[0,1] = 2
F0[1,1] = 1

#Solution process
print(F0)
t=math.atan2(F0[0,1]-F0[1,0],F0[0,0]+F0[1,1])
print(t*180/math.pi)
print((F0[0,1]-F0[1,0])*math.sin(t)+(F0[0,0]+F0[1,1])*math.cos(t))
print(math.sqrt((F0[0,1]-F0[1,0])**2+(F0[0,0]+F0[1,1])**2))

#Printing the corresponding rotation vectors
Q[:,0] = [math.cos(t), math.sin(t), 0]
Q[:,1] = [-math.sin(t), math.cos(t), 0]
Q[:,2] = np.cross(Q[:,0],Q[:,1])
print(Q)
print(np.linalg.det(Q))



