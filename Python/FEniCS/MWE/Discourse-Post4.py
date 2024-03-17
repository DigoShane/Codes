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
gamma = float(0.1) # Learning rate.
NN = int(input('Number of iterations?\n')) # Learning rate.

u = Function(V) 
u_up = Function(V)

phi = (1-u)**2  

# Total potential energy
Pi = phi*dx

#Defining derivative
F = derivative(Pi,u) 
#F_vec = assemble(F)


#Initializing the intial conditions
u0 = interpolate( Expression("x[0]/L",L=L, degree=1), V)#initial cond.
u_up.vector()[:] = u0.vector()[:]


for t in range(NN):
 F_vec = assemble(F)
 #print(F_vec.get_local())
 u.vector()[:] = u_up.vector()[:]
 u_up.vector()[:] = u.vector()[:] - gamma*F_vec[:]
 

plot(u0)
plt.title(r"$u_{0}(x)$",fontsize=26)
plt.show()
plot(u)
plt.title(r"$u(x)$",fontsize=26)
plt.show()



