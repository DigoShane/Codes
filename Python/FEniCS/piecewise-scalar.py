#================================================================================================================
#              Reason for this code
#================================================================================================================
#The reason for this code is to understand how to define piecewise functions.
#This is a beam loaded with a piecewise constant load.

from dolfin import *
import matplotlib.pyplot as plt
import numpy as np


# Create mesh and define function space
mesh = UnitSquareMesh(32, 32)
V = FunctionSpace(mesh, "Lagrange", 1)

# Define Dirichlet boundary (x = 0 or x = 1)
def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)

# Define Function space
u = TrialFunction(V)
v = TestFunction(V)

#Defining the source terms
rho_g1 = Constant(1e-3)
p1 = Expression("x[0]", degree=1, domain=mesh)
p2 = Expression("1-x[0]*x[1]", degree=1, domain=mesh)
p3 = Expression("a*x[1]", a=rho_g1, degree=1, domain=mesh)
#f = Expression('x[0] < 0.25 + DOLFIN_EPS ? p1 : (x[0] < 0.5 + DOLFIN_EPS ? p2 : p3)',
#               p1=p1, p2=p2, p3=p3, degree=2)
f = Expression('x[0] <= 0.25 + DOLFIN_EPS ? p1 : p3', p1=p1, p3=p3, degree=2)
#f = conditional( le('x[0]',0.25), p1, p2)
g = Expression("sin(5*x[0])", degree=2)
a = inner(grad(u), grad(v))*dx
L = f*v*dx + g*v*ds

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Plot solution
plot(interpolate(f, V))
plt.title(r"$f(x)$",fontsize=26)
plt.show()
plot(u)
plt.title(r"$u(x)$",fontsize=26)
plt.show()


