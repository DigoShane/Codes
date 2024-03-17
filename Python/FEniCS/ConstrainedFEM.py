#This code helps teach us how to incorporate integral constraints in FEniCS
#For this we use the Poisson Equation with Neumann BC. 
#In order to fix the solution, the standard approach is to set
# ∫u dx=0
#The eqn is
#-∆u = f in Ω
#∂u/∂n = g on ∂Ω
# ∫u dx=0
# check "https://fenicsproject.org/olddocs/dolfin/1.5.0/python/demo/documented/neumann-poisson/python/documentation.html#:~:text=For%20a%20domain%20%CE%A9%E2%8A%82,c%20by%20the%20above%20equations." for details on the math.

from dolfin import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt

# Create mesh and define function space
mesh = UnitSquareMesh.create(64, 64, CellType.Type.triangle)

# Build function space with Lagrange multiplier
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
R = FiniteElement("Real", mesh.ufl_cell(), 0)
W = FunctionSpace(mesh, P1 * R)

#V = FiniteElement("CG", mesh.ufl_cell(), 2) 
#R = FiniteElement("Real", mesh.ufl_cell(), 0) 
#W = FunctionSpace(mesh, MixedElement(V, R))


# Define variational problem
(u, c) = TrialFunction(W)
(v, d) = TestFunctions(W)
f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2)
g = Expression("-sin(5*x[0])", degree=2)
a = (inner(grad(u), grad(v)) + c*v + u*d)*dx
L = f*v*dx + g*v*ds

# Compute solution
w = Function(W)
solve(a == L, w)
(u, c) = w.split()

# Plot solution
plot(u) 
plt.show()

## Save solution in VTK format
#file = File("neumann_poisson.pvd")
#file << u


