#This file is used to write the output of Nonlinear Solve into a .csv file.


from dolfin import *
from ufl import tanh
import fenics as fe
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.interpolate import interp1d, interp2d

#This writes the output to 
def write2(Vcoord,A,u,file):
 coords = Vcoord.tabulate_dof_coordinates()
 #print(coords[:,0])
 #print("----------------------------------------")
 vecA = A.vector().get_local()
 #print(vecA)
 #print("----------------------------------------")
 vecu = u.vector().get_local()
 #print(vecu)
 #print("----------------------------------------")
 #outfile = open("output.csv", "w")
 outfile = open(file, "w")
 for coord, val1, val2 in zip(coords, vecA, vecu):
    print('{:24.18f}'.format(coord[0]), '{:24.18f}'.format(val1), '{:24.18f}'.format(val2), file=outfile)#Storing (x, A(x), u(x))

def write2Const(Vcoord,A,u,r,file):
 coords = Vcoord.tabulate_dof_coordinates()
 vecA = A.vector().get_local()
 vecu = u.vector().get_local()
 vecr = r.vector().get_local()
 outfile = open(file, "w")
 for coord, val1, val2 in zip(coords, vecA, vecu):
    print('{:24.18f}'.format(coord[0]), '{:24.18f}'.format(val1), '{:24.18f}'.format(val2), '{:24.18f}'.format(vecr[0]), file=outfile)#Storing (x, A(x), u(x), r)
#the r is a constant, so we treat it like a function that is constant on the entire domain.



class ExpressionFromScipyFunction(Expression):
 def __init__(self, f, *args, **kwargs):
  self._f = f
  UserExpression.__init__(self, **kwargs)
 def eval(self, values, x):
  values[:] = self._f(*x)


def read4m(file, V):
 data2 = np.loadtxt(file)
 y0, values1, values2 = data2[:,0], data2[:,1], data2[:,2]
 values = np.transpose( np.column_stack((values1, values2)) )
 interpolant1 = interp1d(y0, values, kind='linear', copy=False, bounds_error=True)
 expression1 = ExpressionFromScipyFunction(interpolant1, element=V.ufl_element())
 return interpolate(expression1, V)

def read4mConst(file, V):
 data2 = np.loadtxt(file)
 y0, values1, values2, value3 = data2[:,0], data2[:,1], data2[:,2], data2[-1,3]
 values = np.transpose( np.column_stack((values1, values2)) )
 interpolant1 = interp1d(y0, values, kind='linear', copy=False, bounds_error=True)
 expression1 = ExpressionFromScipyFunction(interpolant1, element=V.ufl_element())
 return interpolate(expression1, V), value3



def printout(file, V):
 data2 = np.loadtxt(file)
 y0, values1, values2 = data2[:,0], data2[:,1], data2[:,2]
 interpolant1 = interp1d(y0, values1, kind='linear', copy=False, bounds_error=True)
 interpolant2 = interp1d(y0, values2, kind='linear', copy=False, bounds_error=True)
 expression1 = ExpressionFromScipyFunction(interpolant1, element=V.ufl_element())
 expression2 = ExpressionFromScipyFunction(interpolant2, element=V.ufl_element())
 plot(expression1) 
 plt.show()
 plot(expression2) 
 plt.show()


