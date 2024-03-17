#The purpose of this file is to read in a mesh and extend it to a larger domain.
#The meshes are written in .xdmf format.
#We shall use the output of GinzburgLandau-1D-Consraint.py to generate .xdmf files.


from dolfin import *
import fenics as fe
from fenics import XDMFFile, FunctionSpace

# Load the existing XDMF file
fileu = XDMFFile("test-0-Constraint.xdmf") #This is the file in u
fileA = XDMFFile("test-1-Constraint.xdmf") #This is the file in A

# Extract mesh and function space
Lx=100
mesh = fe.IntervalMesh(1000,0,Lx)
V = FunctionSpace(mesh, "Lagrange", 2)#This is for ExtFile

# Create a Function to store the existing solution
u_old = Function(V)
A_old = Function(V)
fileu.read_checkpoint(u_old, "u", 0)
fileA.read_checkpoint(A_old, "A", 0)


plot(u_old)
plt.title(r"$u_old(x)$",fontsize=26)
plt.show()
plot(A_old)
plt.title(r"$A_old(x)e_2$",fontsize=26)
plt.show()


# Create a larger mesh (adjust as needed)
L_new = 200  # New length
mesh_new = fe.IntervalMesh(2000, 0, L_new) #Make sure that this increasing the length means increasing the discretization.

# Create a new FunctionSpace on the larger mesh
V_new = FunctionSpace(mesh_new, "Lagrange", 2) #Try to be consistent with the Element space.

# Interpolate the existing solution onto the new mesh
u_new = interpolate(u_old, V_new)
A_new = interpolate(A_old, V_new)


plot(u_new)
plt.title(r"$u_new(x)$",fontsize=26)
plt.show()
plot(A_new)
plt.title(r"$A_new(x)e_2$",fontsize=26)
plt.show()



