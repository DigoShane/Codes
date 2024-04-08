#The idea here is to run a poisson solver,
# where we read input from a file.

from dolfin import *
from matplotlib import pyplot


import sys
if len(sys.argv) == 0 :
 print("U need to give the name of the input file as an argument. Run Python3 ReadInput.py <inputfile name>")
 exit() 
print ('argument list', sys.argv)
filename = sys.argv[1]

import json
myfile = open( filename, 'r')
jsondata = myfile.read()
obj = json.loads(jsondata)
Nx = int(obj['Nx'])
Ny = int(obj['Ny'])
pord1 = int(obj['pord1'])
pord2 = int(obj['pord2'])
Lx = float(obj['Lx'])
Ly = float(obj['Ly'])
print([Nx, Ny, pord1, pord2, Lx, Ly])

# Create mesh and define function space
mesh = RectangleMesh(Point(0,0) , Point(Lx, Ly), Nx, Ny)
V = FunctionSpace(mesh, "Lagrange", pord1)

# Define Dirichlet boundary (x = 0 or x = 1)
def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f=Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2)
g = Expression("sin(5*x[0])", degree = pord2)
a = inner(grad(u), grad(v))*dx
L = f*v*dx + g*v*ds

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Plot solution
plot(u)
pyplot.show()
