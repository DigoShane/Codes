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
#6. Implement Nesterov acceleration, momentum and Noisy Gradient Descent.
#======================================================================================================
#ISSUES WITH THE CODE:-



from dolfin import *
import fenics as fe
import numpy as np
from ufl import tanh
import matplotlib.pyplot as plt
import mshr


#Create mesh and define function space
lx = 10
ly = 10
kappa = Constqnt(2.0)
domain = mshr.Rectangle(Point(0,0), Point(lx, ly)) 
mesh = mshr.generate_mesh(domain, int(lx*10/kappa), int(ly*10/kappa))
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
tol_abs_in = input("absolute tolerance? ")
tol_abs = Constant(tol_abs_in);
Ae = H*x[0] #The vec pot is A(x) = Hx_1e_2


def curl(a1,a2):
    return a1.dx(0) - a2.dx(1)

#Defining the energy
Pi = ((1-u**2)**2/2 + (1/kappa**2)*inner(grad(u), grad(u)) + ( (a1-t.dx(0))**2 + (a2-t.dx(1))**2 )*u**2 + inner( curl(a1 ,a2-Ae), curl(a1 ,a2-Ae) ) )*dx





#---------------------------------------------------------------------------------------------------------------
##Reading input from a .xdmf file.
#a1a2u = Function(V)
#a1 = Function(Vcoord)
#a2 = Function(Vcoord)
#u = Function(RFnSp)
#a1_in =  XDMFFile("GL-0.xdmf")
#a1_in.read_checkpoint(a1,"a1",0)
#a2_in =  XDMFFile("GL-1.xdmf")
#a2_in.read_checkpoint(a2,"a2",0)
#u_in =  XDMFFile("GL-2.xdmf")
#u_in.read_checkpoint(u,"u",0)
#assign(a1a2u,[a1,a2,u])
##plot(u)
##plt.title(r"$u(x)-b4$",fontsize=26)
##plt.show()
##plot(A)
##plt.title(r"$A(x)e_2-b4$",fontsize=26)
##plt.show()

(a1, a2, u) = split(a1a2u)


F = (-(1-u**2)*u*du + (1/kappa**2)*inner(grad(u), grad(du)) + (a1**2+a2**2)*u*du + u**2*(a1*da1+a2*da2) + inner(curl(a1,a2), curl(da1,da2)))*dx + H*(-da2*ds(5)-da1*ds(4)+da2*ds(3)+da1*ds(2))
solve(F == 0, a1a2u, bcs,
   solver_parameters={"newton_solver":{"convergence_criterion":"residual","relaxation_parameter":rlx_par,"relative_tolerance":0.000001,"absolute_tolerance":tol_abs,"maximum_iterations":10}})

a1 = a1a2u.sub(0, deepcopy=True)
a2 = a1a2u.sub(1, deepcopy=True)
u = a1a2u.sub(2, deepcopy=True)


###Save solution in a .xdmf file
#a1a2u_split = a1a2u.split(True)
#a1a2u_out = XDMFFile('GL-0.xdmf')
#a1a2u_out.write_checkpoint(a1a2u_split[0], "a1", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
#a1a2u_out.close()
#a1a2u_out = XDMFFile('GL-1.xdmf')
#a1a2u_out.write_checkpoint(a1a2u_split[1], "a2", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
#a1a2u_out.close()
#a1a2u_out = XDMFFile('GL-2.xdmf')
#a1a2u_out.write_checkpoint(a1a2u_split[2], "u", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
#a1a2u_out.close()


pie = assemble((1/(4*l*l))*((1-u**2)**2/2 + (1/kappa**2)*inner(grad(u), grad(u)) + (a1**2+a2**2)*u**2 + inner(curl(a1 ,a2-Ae), curl(a1 ,a2-Ae) ))*dx )
#divide by volume, 4l^2-πr^2.
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




