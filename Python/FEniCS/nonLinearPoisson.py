# Warning: from fenics import * will import both `sym` and
# `q` from FEniCS. We therefore import FEniCS first and then
# overwrite these objects.
from fenics import *
import matplotlib.pyplot as plt
import numpy as np

def q(u):
    "Return nonlinear coefficient"
    return 1 + u**2

# Use SymPy to compute f from the manufactured solution u
import sympy as sym
x, y = sym.symbols('x[0], x[1]')
f = - 10*(1 + x + 2*y) #Getting f using the manufactured solution u= 1+x+2y
f = sym.simplify(f)
#u_code = sym.printing.ccode(u)
f_code = sym.printing.ccode(f)
#print('u =', u_code)
print('f =', f_code)

# Create mesh and define function space
#mesh = UnitSquareMesh(8, 8)
#V = FunctionSpace(mesh, 'P', 1)
mesh = UnitSquareMesh(256, 256)#this is for case 2
V = FunctionSpace(mesh, "CG",2)#this is for case 2

# Define boundary condition
u_D = Expression('1+x[0]+2*x[1]', degree=2)#Bc using the manufactured solution u=1+x+2y

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

# Define variational problem
u = Function(V)  # Note: not TrialFunction!
v = TestFunction(V)
f = Expression(f_code, degree=2)
F = q(u)*dot(grad(u), grad(v))*dx - f*v*dx

# Compute solution
solve(F == 0, u, bc)

# Plot solution
plot(u)

# Compute maximum error at vertices. This computation illustrates
# an alternative to using compute_vertex_values as in poisson.py.
u_e = interpolate(u_D, V)
error_max = np.abs(u_e.vector().get_local() - u.vector().get_local()).max()
print('error_max = ', error_max)

# Hold plot
plt.show()
