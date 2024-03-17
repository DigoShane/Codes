"""This program solves Poisson's equation

    - div grad u(x, y,z) = f 
on the unit square with source f given by

1.     f(x, y, z) = Sin(2*Pi*x/l)

and boundary conditions given by

    u(x, y) = 0        for x = 0 or x = 1
du/dn(x, y) = sin(5*x) for y = 0 or y = 1
"""

from dolfin import *
from matplotlib import pyplot

#Parameters needed
l = 0.1 #scaling parameter
L1 = 10 #side of the cube 1
L2 = 20 #side of the cube 2
L3 = 30 #side of the cube 3
f1 = 0.01 
f2 = 0.2
f3 = 0.6
a1 = f1*l
a2 = f2*l
a3 = f3*l
tol = 10**(-2)
print(tol)

#defining the average bulk charge
def BlkChrg:

#Defining the average bulk polarization
def BlkPol:

#Defining the surface charge
def SurChrg:

#Defining the Edge charge
def EdgChrg:

#Defining the corner charge
def CorChrg:

# Create mesh and define function space
mesh = UnitSquareMesh(32, 32)
V = FunctionSpace(mesh, "Lagrange", 1)

# Define Dirichlet boundary (x = 0 or x = 1)
def boundary_D(x):
    return on_boundary and near(x[2], 0, tol)
    #return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Expression("sin[2*Pi*x[0]/l]", degree=2)
g = Constant(0) #Expression("sin(5*x[0])", degree=2)
a = inner(grad(u), grad(v))*dx
L = f*v*dx + g*v*ds

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Save solution in VTK format
file = File("poisson.pvd")
file << u

# Plot solution
plot(u, interactive=True)
pyplot.show()
