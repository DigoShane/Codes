#This is the same code as the prev. Only difference is we want to define 
#the variables as Coefficients here.


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
a1 = Function(Vcoord)
a2 = Function(Vcoord)
t = Function(Vcoord)
u = Function(Vcoord)
a1_up = Function(Vcoord)
a2_up = Function(Vcoord)
t_up = Function(Vcoord)
u_up = Function(Vcoord)

# Parameters
gamma = float(0.01) # Learning rate.
NN = int(input('Number of iterations?\n')) # Number of iterations
Hin = input("External Magnetic field? ")
H = Constant(Hin);
tol_in = input("absolute tolerance? ")
tol = float(tol_in);
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
#A1 = interpolate( Expression("0.0", degree=2), Vcoord)
#A2 = interpolate( Expression("0.0", degree=2), Vcoord)
#T = interpolate( Expression("1.0", degree=2), Vcoord)
#U = interpolate( Expression("1.0", degree=2), Vcoord)
#Normal state
A1 = interpolate( Expression("0.0", degree=2), Vcoord)
A2 = interpolate( Expression("H*x[0]", H=H, degree=2), Vcoord)
T = interpolate( Expression("x[1]", degree=2), Vcoord)
U = interpolate( Expression("x[0]", degree=2), Vcoord)
#Vortex Solution.
#..... need to complete
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
 print(Fa1_vec.get_local()) # prints the vector.
 print(np.linalg.norm(np.asarray(Fa1_vec.get_local()))) # prints the vector's norm.
 tol_test = np.linalg.norm(np.asarray(Fa1_vec.get_local()))\
           +np.linalg.norm(np.asarray(Fa2_vec.get_local()))\
           +np.linalg.norm(np.asarray(Ft_vec.get_local()))\
           +np.linalg.norm(np.asarray(Fu_vec.get_local()))
 print(tol_test)
 if float(tol_test)  < tol :
  break
 

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


pie = assemble((1/(lx*ly))*((1-u**2)**2/2 + (1/kappa**2)*inner(grad(u), grad(u)) \
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


