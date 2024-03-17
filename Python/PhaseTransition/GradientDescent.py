#The objective of this python code is to write an energy minimization method in FEniCS.
#I want to input an energy density on which i ca use gradient descent.
#------------------------------------------------------------------------------------
#The problem is a simple Minimization problem.
#We will modify the functional at each step to make it a bit challenging.
#1.  Inf ∫(u-1)^2 dx --> argmin = 1
#2.  Inf ∫u^2+u'^2 dx --> argmin = 0
#3.  Inf ∫u^2 + (u'-b)^2 dx //This is convex but u' matters unlike the other cases.
#    --> argmin = b sinh(x)/cos 1
#3.  Inf ∫(1-u^2)^2 + (u')^2 dx //This is non convex.
#4.  Inf ∫(1-u^2)^2+ u + (u')^2 dx //This is non convex.
#------------------------------------------------------------------------------------



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
#phi = ( u**2 + inner( ufl.grad(u), ufl.grad(u)) )
phi = ( u**2 + inner( ufl.grad(u - x[0]), ufl.grad(u - x[0]) ) ) # b=1
#phi = ( (1-u**2)**2 + inner( ufl.grad(u), ufl.grad(u)) )
#phi = ( (1-u**2)**2 + u + inner( ufl.grad(u), ufl.grad(u)) )

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

print("The energy density is", assemble(Pi)/L)

plot(u0)
plt.title(r"$u_{0}(x)$",fontsize=26)
plt.show()
plot(u)
plt.title(r"$u(x)$",fontsize=26)
plt.show()




