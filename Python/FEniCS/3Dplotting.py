#the objective of this code is to represent a surface plot. The standard plot due to matplotlib in FEniCS, can't do surface plot.
#This code tells us how we can achieve that.
#This tutorial is taken from "https://fenicsproject.org/pub/tutorial/html/._ftut1020.html"
#---------------------------------------------------------------------------------------------------------
# Solving the heat equation in 2D with T(x_1=0)=T(x_1=1)=0.
#There is a heat source f at (0.5,0.5) with flux boundary conditions (g) at x_2=0 and x_2=1.
#=========================================================================================================
#This is written specificaly for rectangular domains in mind.

import dolfin
print(f"DOLFIN version: {dolfin.__version__}")
from dolfin import *
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # necessary for 3D plotting
from matplotlib import cm
from matplotlib.ticker import LinearLocator


# Create mesh and define function space
N = 64  # Number of discretization points in each direction
mesh = UnitSquareMesh(N, N)
V = FunctionSpace(mesh, "Lagrange", 1)

# Define Dirichlet boundary (x = 0 or x = 1)
def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2)
g = Expression("sin(5*x[0])", degree=2)
a = inner(grad(u), grad(v))*dx
L = f*v*dx + g*v*ds

# Compute solution
u = Function(V)
solve(a == L, u, bc)


# Std FEniCS plot
plot(u, interactive=True)
plt.title(r"$u(x)$", fontsize=26)
plt.show()


#saving uh to XDMF file 
xdmf_file = XDMFFile("output.xdmf") 
xdmf_file.write(u, 0) 
#saving uh to ParaView file 
pvd_file = File("output.pvd") 
pvd_file << u


xplt = V.tabulate_dof_coordinates()
Z = np.reshape(u.vector()[:], (-1, N+1))

v2d = vertex_to_dof_map(V)
nodal_values = u.vector()[:]


#Surface plot
X = np.linspace(0, 1, N+1)
Y = np.linspace(0, 1, N+1)
X, Y = np.meshgrid(X, Y)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

plt.xlabel("x")
plt.ylabel("y")
plt.show()
