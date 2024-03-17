#The objective of this python code is to write an Gradient Descent method for the Ginzburg Landau equations.
#


import dolfin
print(f"DOLFIN version: {dolfin.__version__}")
from dolfin import *
import ufl
print(f" UFL version: {ufl.__version__}")
import matplotlib.pyplot as plt
from petsc4py import PETSc

L=10
mesh = IntervalMesh(100,0,L)
x = SpatialCoordinate(mesh)
element = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
V = FunctionSpace(mesh, "CG", 1)
gamma = float(0.01)# Learning rate.
tol = float(input('tolerance?\n')) # Learning rate.
NN = int(input('Number of iterations?\n')) # Learning rate.

u = Function(V) 
u_up = Function(V)

#phi = (1-u)**2  
phi = ( u**2 + inner( ufl.grad(u), ufl.grad(u)) )

# Total potential energy
Pi = phi*dx

#Defining derivative
F = derivative(Pi,u) 


#Initializing the intial conditions
u0 = interpolate( Expression("x[0]/L",L=L, degree=1), V)#initial cond.
u_up.vector()[:] = u0.vector()[:]


for t in range(NN):
 u.vector()[:] = u_up.vector()[:]
 F_vec = assemble(F)
 u_up.vector()[:] = u.vector()[:] - gamma*F_vec[:]
 if (norm(F_vec)) < tol:
  break
 #print(F_vec.get_local()) # prints the vector.
 
 
print(F_vec.get_local()) # prints the vector.

plot(u0)
plt.title(r"$u_{0}(x)$",fontsize=26)
plt.show()
plot(u)
plt.title(r"$u(x)$",fontsize=26)
plt.show()




