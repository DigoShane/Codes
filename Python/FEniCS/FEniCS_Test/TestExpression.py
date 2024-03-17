"""The objective here is to test 
how accurately can FEniCS represent 
an expression
"""

from dolfin import *
from matplotlib import pyplot
import numpy as np

#Parameters needed
ell = 0.05 #scaling parameter
L1 = 2 #length
tol = 10**(-1)

# Create mesh and define function space
mesh = UnitSquareMesh(128, 128)
V = FunctionSpace(mesh, "Lagrange", 2)

# Define variational problem
f = interpolate(Expression('sin(2*Pie*x[0]/l)/l', degree=1, l=ell, Pie=np.pi), V)

# Plot solution
plot(f)
pyplot.show()
