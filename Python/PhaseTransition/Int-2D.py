#The objective is to test out Numerical integration in 2D


import numpy as np
from scipy import special as sp
from scipy import integrate
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import math
import sys #for error messages
from scipy.fft import fft2, rfft2, irfft2
from scipy import integrate



def double_Integral(xmin, xmax, ymin, ymax, nx, ny, A):

    dS = ((xmax-xmin)/(nx-1)) * ((ymax-ymin)/(ny-1))

    A_Internal = A[1:-1, 1:-1]

    # sides: up, down, left, right
    (A_u, A_d, A_l, A_r) = (A[0, 1:-1], A[-1, 1:-1], A[1:-1, 0], A[1:-1, -1])

    # corners
    (A_ul, A_ur, A_dl, A_dr) = (A[0, 0], A[0, -1], A[-1, 0], A[-1, -1])

    return dS * (np.sum(A_Internal)\
                + 0.5 * (np.sum(A_u) + np.sum(A_d) + np.sum(A_l) + np.sum(A_r))\
                + 0.25 * (A_ul + A_ur + A_dl + A_dr))

A = [[0.0, 0.209301, 0.390687, 0.529304, 0.63031, 0.716463, 0.817469, 0.956086, 1.13747], [0.0748208, 0.270164, 0.444124, 0.58532, 0.697704, 0.798711, 0.911096, 1.05229, 1.22625], [0.149642, 0.309643, 0.464797, 0.612524, 0.75372, 0.892337, 1.03353, 1.18126, 1.33641], [0.224462, 0.344274, 0.478043, 0.633197, 0.807156, 0.988542, 1.1625, 1.31766, 1.45142], [0.299283, 0.392863, 0.512674, 0.672675, 0.868019, 1.07732, 1.27266, 1.43267, 1.55248], [0.374104, 0.467683, 0.587495, 0.747496, 0.94284, 1.15214, 1.34749, 1.50749, 1.6273], [0.448925, 0.568736, 0.702505, 0.857659, 1.03162, 1.213, 1.38696, 1.54212, 1.67589], [0.523745, 0.683747, 0.838901, 0.986628, 1.12782, 1.26644, 1.40764, 1.55536, 1.71052], [0.598566, 0.79391, 0.967869, 1.10907, 1.22145, 1.32246, 1.43484, 1.57604, 1.75]]

print("--------------------------------------------------------")
print("A^2 =")
print('\n'.join([''.join(['{:25}'.format(item) for item in row]) 
      for row in np.around( np.power(A,2),4)]))


print( double_Integral(0,1,0,1, 9, 9, np.asarray( np.power(A,1) ) ) )
