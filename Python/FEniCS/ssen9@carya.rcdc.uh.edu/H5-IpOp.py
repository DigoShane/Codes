#The objective is to implement input output using the HDF5 format
#This should be more robust than writing the output to a .csv file
# or allow for mixed elements unlike the XDMF file.
#-----------------------------------------------------------------
#I will achieve this using the simple Poisson equation example. 
#I will read f from a file "xdmf-input.xdmf" and write u to "xdmf-output.xdmf"
#The output of method 1 is stored in "xdmf-output.xdmf" and then we change
#the name to "xdmf-input.xdmf" and run method 2.
#The equation to solve is:-
#Ω=[0,1]
#-∆u=f* in Ω
#u =0 on ∂Ω
#f= L(f* )
#----------------------------------------------------------------
#where f* is defined as
#-∆f*=f in Ω
# f* =0 on ∂Ω
#f= exp(-x^2)


from fenics import *
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.interpolate import interp1d, interp2d

parameters["allow_extrapolation"]=True

# Create mesh and define function space
mesh = UnitIntervalMesh(32)
pord=1;
V = FunctionSpace(mesh, "Lagrange", pord)

# Define Dirichlet boundary (x = 0 or x = 1)
def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)

# Defining the load
##method1
#f = Expression("sin(x[0]*l)",l=7, degree=pord)
#method2
f = Function(V)
f_in =  XDMFFile("test.xdmf")
f_in.read_checkpoint(f,"u",0)

# Defining a and L
a = inner(grad(u), grad(v))*dx
L = f*v*dx

# Compute solution
u = Function(V)
solve(a == L, u, bc)

#Save solution in a .csv file
f_out = XDMFFile('test.xdmf')
f_out.write_checkpoint(project(u,V), "u", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
f_out.close()


# Plot solution
plot(u)
plt.show()


