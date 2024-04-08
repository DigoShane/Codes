#The objective of this code is to test how to use Fn Assigner.
#We will do this by reading in a Text file, creating a function and then cimbining them using FnAssigner


from dolfin import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.interpolate import interp1d, interp2d


class ExpressionFromScipyFunction(Expression):
 def __init__(self, f, *args, **kwargs):
  self._f = f
  UserExpression.__init__(self, **kwargs)
 def eval(self, values, x):
  values[:] = self._f(*x)


# Create mesh and define function space
mesh = UnitIntervalMesh(10)

# Build function space with Lagrange multiplier
#Vcoord = FunctionSpace(mesh, "Lagrange", 1)#We use this for read & write using ExtFile.py
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
R = FiniteElement("Real", mesh.ufl_cell(), 0)
W = FunctionSpace(mesh, P1 * R)

#V = FiniteElement("CG", mesh.ufl_cell(), 2) 
#R = FiniteElement("Real", mesh.ufl_cell(), 0) 
#W = FunctionSpace(mesh, MixedElement(V, R))


# Define variational problem
(u, c) = TrialFunction(W)
(v, d) = TestFunctions(W)



data = np.loadtxt('read.txt')
y0, values1, values2 = data[:,0], data[:,1], data[:,2]
values = np.transpose( np.column_stack((values1, values2)) )
interpolant1 = interp1d(y0, values, kind='linear', copy=False, bounds_error=True)
expression1 = ExpressionFromScipyFunction(interpolant1, element=W.ufl_element())
Au = interpolate(expression1, W)
print("here")

(A,u) = split(Au)

print("------------------------------")
print(y0)
print("------------------------------")
print(values1)
print("------------------------------")
print(values2)



# Plot solution
plot(u)
plt.title("$u(x)$",fontsize=26)
plt.show()
plot(A)
plt.title("$A(x)e_2$",fontsize=26)
plt.show()

