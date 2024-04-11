#Here we solve the 2D Ginzbug Landau problem with an applied magnetic field.
#Here we want to use Energy minimization method. We start off with Gradient Descent.
#HEre a1 is \ve{A}\cdot e_1, a2 is \ve{A}\cdot e_2, u is u. However, \theta=t
#------------------------------------------------------------------------------------------------------
# For details on progress, visit the overleaf file:-
#1. Overleaf. superconductivity-Pradeep+Liping/Z3-Coding.tex/Sec. Stochastic Energy minimization methods
# /subsec. Gradient Descent in FEniCS/paragraph{Wrote a 2D Ginzburg LAndau Energy minimization code}
#======================================================================================================
#The way the Code works
#1. The input to the code is:
#   a. The external field
#   b. The relaxation parameter
#   c. The absolute tolerance
#2. When reading from and writing into respective files,
#   we are writing the lagrange multiplier as a constant function
#   When reading the functions, we interpolate onto a space VAu.
#IMPORTANT:-
# At each step we do $\theta\mapsto\theta mod 2\pi$.
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

import time # timing for performance test.
t0 = time.time()

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

import sys
np.set_printoptions(threshold=sys.maxsize)

#Parameters
print("================input to code========================")
kappa = Constant(2.0)
lx = float(input("lx? --> "))
ly = float(input("ly? --> "))
gamma = float(input('Learning rate? -->')) # Learning rate.
NN = int(input('Number of iterations? -->')) # Number of iterations
H = Constant(input("External Magnetic field? -->"));
tol = float(input("absolute tolerance? --> "))
read_in = int(input("Read from file? 1 for Yes, 0 for No --> "))

#Create mesh and define function space
mesh = RectangleMesh(Point(0., 0.), Point(lx, ly), np.ceil(lx*10/kappa), np.ceil(ly*10/kappa), "crossed") # "crossed means that first it will partition the ddomain into Nx x Ny rectangles. Then each rectangle is divided into 4 triangles forming a cross"
x = SpatialCoordinate(mesh)
Ae = H*x[0] #The vec pot is A(x) = Hx_1e_2
V = FunctionSpace(mesh, "Lagrange", 2)#This is for ExtFile

# Define functions
a1 = Function(V)
a2 = Function(V)
t = Function(V)
u = Function(V)
a1_up = Function(V)
a2_up = Function(V)
t_up = Function(V)
u_up = Function(V)

def curl(a1,a2):
    return a2.dx(0) - a1.dx(1)

#Defining the energy
Pi = ( (1-u**2)**2/2 + (1/kappa**2)*inner(grad(u), grad(u)) + ( (a1-t.dx(0))**2 + (a2-t.dx(1))**2 )*u**2 \
      + inner( curl(a1 ,a2-Ae), curl(a1 ,a2-Ae) ) )*dx

#Defining the gradient
Fa1 = derivative(Pi, a1)
Fa2 = derivative(Pi, a2)
Ft = derivative(Pi, t)
Fu = derivative(Pi, u)


##Setting up the initial conditions
if read_in == 0: # We want to use the standard values.
 ##SC state
 #print("Using bulk SC as initial condition")
 #A1 = interpolate( Expression("0.0", degree=2), V)
 #A2 = interpolate( Expression("0.0", degree=2), V)
 #T = interpolate( Expression("1.0", degree=2), V)
 #U = interpolate( Expression("1.0", degree=2), V)
 ##Modified normal state
 #print("Using modified bulk Normal as initial condition")
 #A1 = interpolate( Expression("0.0", degree=2), V)
 #A2 = interpolate( Expression("H*x[0]", H=H, degree=2), V)
 #T = interpolate( Expression("x[1]", degree=2), V)
 #U = interpolate( Expression("x[0]", degree=2), V)
 ##Vortex Solution.
 print("Using Vortex solution")
 A1 = interpolate( Expression('sqrt((x[0]-0.5*lx)*(x[0]-0.5*lx)+(x[1]-0.5*ly)*(x[1]-0.5*ly)) <= r + DOLFIN_EPS ? -x[1] : \
                             -exp(-sqrt((x[0]-0.5*lx)*(x[0]-0.5*lx)+(x[1]-0.5*ly)*(x[1]-0.5*ly))) \
                              *x[1]/sqrt((x[0]-0.5*lx)*(x[0]-0.5*lx)+(x[1]-0.5*ly)*(x[1]-0.5*ly))*1/K', \
                                lx=lx, ly=ly, r=0.3517, K=kappa, degree=2), V)
 A2 = interpolate( Expression('sqrt((x[0]-0.5*lx)*(x[0]-0.5*lx)+(x[1]-0.5*ly)*(x[1]-0.5*ly)) <= r + DOLFIN_EPS ? x[0] : \
                             exp(-sqrt((x[0]-0.5*lx)*(x[0]-0.5*lx)+(x[1]-0.5*ly)*(x[1]-0.5*ly))) \
                              *x[0]/sqrt((x[0]-0.5*lx)*(x[0]-0.5*lx)+(x[1]-0.5*ly)*(x[1]-0.5*ly))*1/K', \
                                lx=lx, ly=ly, r=0.3517, K=kappa, degree=2), V)
 T = interpolate( Expression('(x[0]-0.5*lx)*(x[0]-0.5*lx)+(x[1]-0.5*ly)*(x[1]-0.5*ly) <= r*r + DOLFIN_EPS ? 1 \
                             : atan2(x[0]-0.5*lx,x[1]-0.5*ly)', lx=lx, ly=ly, r=0.001, degree=2), V)
 U = interpolate( Expression('tanh(sqrt((x[0]-0.5*lx)*(x[0]-0.5*lx)+(x[1]-0.5*ly)*(x[1]-0.5*ly)))', lx=lx, ly=ly, degree=2), V) 
 #..... need to complete
###---------------------------------------------------------------------------------------------------------------
elif read_in == 1: # We want to read from xdmf files
 #Reading input from a .xdmf file.
 print("reading in previous output as initial condition.")
 A1 = Function(V)
 A2 = Function(V)
 T = Function(V)
 U = Function(V)
 a1_in =  XDMFFile("GL-2DEnrg-0.xdmf")
 a1_in.read_checkpoint(A1,"a1",0)
 a2_in =  XDMFFile("GL-2DEnrg-1.xdmf")
 a2_in.read_checkpoint(A2,"a2",0)
 t_in =  XDMFFile("GL-2DEnrg-2.xdmf")
 t_in.read_checkpoint(T,"t",0)
 u_in =  XDMFFile("GL-2DEnrg-3.xdmf")
 u_in.read_checkpoint(U,"u",0)
 #plot(u)
 #plt.title(r"$u(x)-b4$",fontsize=26)
 #plt.show()
else:
 import sys
 sys.exit("Not a valid input for read_in.")

a1_up.vector()[:] = A1.vector()[:]
a2_up.vector()[:] = A2.vector()[:]
t_up.vector()[:] = T.vector()[:]
u_up.vector()[:] = U.vector()[:]

for tt in range(NN):
 a1.vector()[:] = a1_up.vector()[:]
 a2.vector()[:] = a2_up.vector()[:]
 t.vector()[:] = t_up.vector()[:] % (2*np.pi) # doing mod 2\pi
 if any(t.vector()[:] < -np.pi) or any(t.vector()[:] > np.pi):
  print("============================================================")
  print("before modding the previous output")
  print(t_up.vector()[:])
  print("============================================================")
  print("after modding the previous output")
  print(t.vector()[:])
 u.vector()[:] = u_up.vector()[:]
 Fa1_vec = assemble(Fa1)
 Fa2_vec = assemble(Fa2)
 Ft_vec = assemble(Ft)
 Fu_vec = assemble(Fu)
 a1_up.vector()[:] = a1.vector()[:] - gamma*Fa1_vec[:]
 a2_up.vector()[:] = a2.vector()[:] - gamma*Fa2_vec[:]
 t_up.vector()[:] = t.vector()[:] - gamma*Ft_vec[:]
 u_up.vector()[:] = u.vector()[:] - gamma*Fu_vec[:]
 #print(Fa1_vec.get_local()) # prints the vector.
 #print(np.linalg.norm(np.asarray(Fa1_vec.get_local()))) # prints the vector's norm.
 tol_test = np.linalg.norm(np.asarray(Fa1_vec.get_local()))\
           +np.linalg.norm(np.asarray(Fa2_vec.get_local()))\
           +np.linalg.norm(np.asarray(Ft_vec.get_local()))\
           +np.linalg.norm(np.asarray(Fu_vec.get_local()))
 #print(tol_test)
 if float(tol_test)  < tol :
  break
 

##Save solution in a .xdmf file and for paraview.
a1a2tu_out = XDMFFile('GL-2DEnrg-0.xdmf')
a1a2tu_out.write_checkpoint(a1, "a1", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("GL-2DEnrg-0.pvd") # for paraview. 
pvd_file << a1
a1a2tu_out.close()
a1a2tu_out = XDMFFile('GL-2DEnrg-1.xdmf')
a1a2tu_out.write_checkpoint(a2, "a2", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("GL-2DEnrg-1.pvd") # for paraview. 
pvd_file << a2
a1a2tu_out.close()
a1a2tu_out = XDMFFile('GL-2DEnrg-2.xdmf')
a1a2tu_out.write_checkpoint(t, "t", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("GL-2DEnrg-2.pvd") # for paraview.
pvd_file << t
a1a2tu_out.close()
a1a2tu_out = XDMFFile('GL-2DEnrg-3.xdmf')
a1a2tu_out.write_checkpoint(u, "u", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("GL-2DEnrg-3.pvd") 
pvd_file << u
a1a2tu_out.close()


pie = assemble((1/(lx*ly))*((1-u**2)**2/2 + (1/kappa**2)*inner(grad(u), grad(u)) \
                        + ( (a1-t.dx(0))**2 + (a2-t.dx(1))**2 )*u**2 \
                            + inner( curl(a1 ,a2-Ae), curl(a1 ,a2-Ae) ) )*dx )
print("================output of code========================")
print("Energy density is", pie)
print("gamma = ", gamma)
print("kappa = ", kappa)
print("lx = ", lx)
print("ly = ", ly)
print("NN = ", NN)
print("H = ", H)
print("tol = ", tol, ", ", float(tol_test))
print("read_in = ", read_in)


c = plot(u)
plt.title(r"$u(x)$",fontsize=26)
plt.colorbar(c)
plt.show()
c = plot(a1)
plt.title(r"$A_1(x)$",fontsize=26)
plt.colorbar(c)
plt.show()
c = plot(a2)
plt.title(r"$A_2(x)$",fontsize=26)
plt.colorbar(c)
plt.show()
c = plot(t)
plt.title(r"$\theta(x)$",fontsize=26)
plt.colorbar(c)
plt.show()

t1 = time.time()

print("time taken for code to run = ", t1-t0)
