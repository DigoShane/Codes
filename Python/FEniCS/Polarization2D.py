"""This program solves Poisson's equation

    - div grad u(x, y) = f 
on the unit square with source f given by

1.     f(x, y) = Sin(2*Pi*x/l)

and boundary conditions given by

    u(x, y) = 0        for y = 0 
    u(x, y) = 10       for y = L2
du/dn(x, y) = 0        otherwise
"""

from dolfin import *
from matplotlib import pyplot
import numpy as np
#import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # necessary for 3D plotting
from matplotlib import cm
from boxfield import *

#Parameters needed
ell = 0.5 #scaling parameter
L1 = 2 #side of the cube 1
L2 = 1 #side of the cube 2
nx = 2048 # discretizations along x
ny = 32 #discretizations along y
f1 = 0.1 
f2 = 0.2
a1 = f1*ell
a2 = f2*ell
tol = 10**(-1)

# Create mesh and define function space
mesh = RectangleMesh(Point(0,0), Point(L1,L2), nx, ny)
V = FunctionSpace(mesh, "Lagrange", 2)

# Define Dirichlet boundary (y = 0 or y = L2)
def boundary_D4(x, on_boundary): #y=0, \gamma4
    return on_boundary and near(x[1], L2, tol)

def boundary_D2(x, on_boundary): #y=L2, \gamma2
    return on_boundary and near(x[1], 0, tol)

# Define boundary condition
u04 = Constant(0.0)
bc4 = DirichletBC(V, u04, boundary_D4)

u02 = Constant(10.0)
bc2 = DirichletBC(V, u02, boundary_D2)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Expression('sin(2*Pie*x[0]/l)/l', degree=1, l=ell, Pie=np.pi)
g = Constant(0) 
a = inner(grad(u), grad(v))*dx
L = f*v*dx + g*v*ds

# Compute solution
u = Function(V)
bc = [bc2, bc4]
solve(a == L, u, bc)

# Save solution in VTK format
file = File("Polarization2D.pvd")
file << u

# Plot solution
#c = plot(u, interactive=True)
#print(type(c))
#print(dir(c))
#pyplot.colorbar(c)
#pyplot.show()

#u_box = structured_mesh(u, (nx, ny))
u_box = FEniCSBoxField(u, (nx, ny))
u_ = u_box.values
fig = plt.figure()
ax = fig.gca(projection='3d')
cv = u_box.grid.coorv  # vectorized mesh coordinates
ax.plot_surface(cv[X], cv[Y], u_, cmap=cm.coolwarm,
                rstride=1, cstride=1)
plt.title('Surface plot of solution')
