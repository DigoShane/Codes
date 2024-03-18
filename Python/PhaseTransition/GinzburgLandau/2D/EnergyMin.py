#Here we solve the 2D Ginzbug Landau problem with an applied magnetic field.
#Here we want to use Energy minimization method. We start off with Gradient Descent.
#HEre a1 is \ve{A}\cdot e_1, a2 is \ve{A}\cdot e_2, u is u. However, \theta=t
#======================================================================================================
#The way the Code works
#1. The input to the code is:
#   a. The external field
#   b. The relaxation parameter
#   c. The absolute tolerance
#2. When reading from and writing into respective files,
#   we are writing the lagrange multiplier as a constant function
#   When reading the functions, we interpolate onto a space VAu.
#======================================================================================================
#Things to keep in mind about writing this code:-
#1. Define a functoon to evaluate the curl
#2. Define a rotation funciton.
#3. HAve replace L with l throught.
#4. All variables are lower case.
#5. REdo the code by using Hn\cdot B\perp
#6. Implement Nesterov acceleration, momentum, minibatch gradient descent and Noisy Gradient Descent.
#7. put in initial conditions for vortex solution.
#======================================================================================================
#ISSUES WITH THE CODE:-


import dolfin
print(f"DOLFIN version: {dolfin.__version__}")
from dolfin import *
import fenics as fe
import numpy as np
import ufl
print(f" UFL version: {ufl.__version__}")
from ufl import tanh
import matplotlib.pyplot as plt
import mshr


#Create mesh and define function space
lx = 10
ly = 10
kappa = Constant(2.0)
mesh = RectangleMesh(Point(0., 0.), Point(lx, ly), np.ceil(lx*10/kappa), np.ceil(ly*10/kappa), "crossed")
x = SpatialCoordinate(mesh)
Va1 = FiniteElement("CG", mesh.ufl_cell(), 2)
Va2 = FiniteElement("CG", mesh.ufl_cell(), 2)
Vt = FiniteElement("CG", mesh.ufl_cell(), 2)
Vu = FiniteElement("CG", mesh.ufl_cell(), 2)
V = FunctionSpace(mesh, MixedElement(Va1, Va2, Vt, Vu))
Vcoord = FunctionSpace(mesh, "Lagrange", 2)#This is for ExtFile


# Define functions
a1a2tu = Function(V)
(a1, a2, t, u) = split(a1a2tu)
a1a2tu_up = Function(V)
(a1_up, a2_up, t_up, u_up) = split(a1a2tu_up)

# Parameters
gamma = float(0.01) # Learning rate.
NN = int(input('Number of iterations?\n')) # Number of iterations
Hin = input("External Magnetic field? ")
H = Constant(Hin);
tol_in = input("absolute tolerance? ")
tol = Constant(tol_in);
Ae = H*x[0] #The vec pot is A(x) = Hx_1e_2


def curl(a1,a2):
    return a1.dx(0) - a2.dx(1)

#Defining the energy
Pi = ((1-u**2)**2/2 + (1/kappa**2)*inner(grad(u), grad(u)) + ( (a1-t.dx(0))**2 + (a2-t.dx(1))**2 )*u**2 \
      + inner( curl(a1 ,a2-Ae), curl(a1 ,a2-Ae) ) )*dx

#Defining the gradient
Fa1 = derivative(Pi, a1)
Fa2 = derivative(Pi, a2)
Ft = derivative(Pi, t)
Fu = derivative(Pi, u)


##Setting up the initial conditions
##SC state
#A1 = Expression("0.0", degree=2)
#A2 = Expression("0.0", degree=2)
#T = Expression("1.0", degree=2)
#U = Expression("1.0", degree=2)
##Normal state
#A1 = Expression("0.0", degree=2)
#A2 = Expression("H*x[0]", H=H, degree=2)
#T = Expression("0.0", degree=2)
#U = Expression("0.0", degree=2)
#PErturbed normal state
A1 = Expression("H*x[1]", degree=2)
A2 = Expression("H*x[0]", H=H, degree=2)
T = Expression("x[1]-x[0]", degree=2)
U = Expression("1.0", degree=2)
#Vortex Solution.
#..... Complete
#---------------------------------------------------------------------------------------------------------------
##Reading input from a .xdmf file.
#a1a2u = Function(V)
#a1 = Function(Vcoord)
#a2 = Function(Vcoord)
#t = Function(Vcoord)
#u = Function(Vcoord)
#a1_in =  XDMFFile("GL-2DEnrg-0.xdmf")
#a1_in.read_checkpoint(a1,"a1",0)
#a2_in =  XDMFFile("GL-2DEnrg-1.xdmf")
#a2_in.read_checkpoint(a2,"a2",0)
#t_in =  XDMFFile("GL-2DEnrg-2.xdmf")
#t_in.read_checkpoint(t,"t",0)
#u_in =  XDMFFile("GL-2DEnrg-3.xdmf")
#u_in.read_checkpoint(u,"u",0)
#assign(a1a2tu,[a1,a2,t,u])
##plot(u)
##plt.title(r"$u(x)-b4$",fontsize=26)
##plt.show()

a1_up.vector()[:] = A1.vector()[:]
a2_up.vector()[:] = A2.vector()[:]
t_up.vector()[:] = T.vector()[:]
u_up.vector()[:] = U.vector()[:]

for tt in range(NN):
 a1.vector()[:] = a1_up.vector()[:]
 a2.vector()[:] = a2_up.vector()[:]
 t.vector()[:] = t_up.vector()[:]
 u.vector()[:] = u_up.vector()[:]
 Fa1_vec = assemble(Fa1)
 Fa2_vec = assemble(Fa2)
 Ft_vec = assemble(Ft)
 Fu_vec = assemble(Fu)
 a1_up.vector()[:] = a1.vector()[:] - gamma*Fa1_vec[:]
 a2_up.vector()[:] = a2.vector()[:] - gamma*Fa2_vec[:]
 t_up.vector()[:] = t.vector()[:] - gamma*Ft_vec[:]
 u_up.vector()[:] = u.vector()[:] - gamma*Fu_vec[:]
 if ( norm(Fa1_vec) + norm(Fa2_vec) + norm(Ft_vec) + norm(Fu_vec) ) < tol:
  break
 #print(F_vec.get_local()) # prints the vector.
 

##Save solution in a .xdmf file
a1a2tu_out = XDMFFile('GL-2DEnrg-0.xdmf')
a1a2tu_out.write_checkpoint(a1, "a1", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
a1a2tu_out.close()
a1a2tu_out = XDMFFile('GL-2DEnrg-1.xdmf')
a1a2tu_out.write_checkpoint(a2, "a2", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
a1a2tu_out.close()
a1a2tu_out = XDMFFile('GL-2DEnrg-2.xdmf')
a1a2tu_out.write_checkpoint(t, "t", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
a1a2tu_out = XDMFFile('GL-2DEnrg-2.xdmf')
a1a2tu_out.write_checkpoint(u, "u", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
a1a2tu_out.close()


pie = assemble((1/(l*l))*((1-u**2)**2/2 + (1/kappa**2)*inner(grad(u), grad(u)) \
                        + ( (a1-t.dx(0))**2 + (a2-t.dx(1))**2 )*u**2 \
                            + inner( curl(a1 ,a2-Ae), curl(a1 ,a2-Ae) ) )*dx )
print("Energy density =", pie)


plot(u)
plt.title(r"$u(x)$",fontsize=26)
plt.show()
plot(a1)
plt.title(r"$A_1(x)$",fontsize=26)
plt.show()
plot(a2)
plt.title(r"$A_2(x)$",fontsize=26)
plt.show()
plot(t)
plt.title(r"$\theta(x)$",fontsize=26)
plt.show()


