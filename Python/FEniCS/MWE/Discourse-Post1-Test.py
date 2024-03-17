#This was the response solution.

from dolfin import *
import fenics as fe
import matplotlib.pyplot as plt
import numpy as np


#Create mesh and define function space
Lx=10.  #length of domain
mesh = fe.IntervalMesh(1000,0,Lx)
x = SpatialCoordinate(mesh)
VA = FiniteElement("CG", mesh.ufl_cell(), 2) #Element for A
Vu = FiniteElement("CG", mesh.ufl_cell(), 2) #Element for u
R = FiniteElement("Real", mesh.ufl_cell(), 0) #Element for Lagrange multiplier
V = FunctionSpace(mesh, MixedElement(VA, Vu, R)) #Creating MixedFunctionSpace
VFnSp = FunctionSpace(mesh, "Lagrange", 2)#FnSpace for u and A
RFnSp = FunctionSpace(mesh, "Real", 0)#Fn space for lagrange multiplier

#Newton Rhapson input
Aur = interpolate( Expression(("1","0.0", "3.5"), degree=2), V)#SC phase as initial cond.

def print_function_info(pre_text, function):
    local = function.vector().get_local()
    print(pre_text, np.min(local), np.max(local))

#(A, u, r) = split(Aur)
#plot(u)
#plt.title(r"$u(x)-b4$",fontsize=26)
#plt.show()
#plot(A)
#plt.title(r"$A(x)e_2-b4$",fontsize=26)
#plt.show()

for deepcopy in (False, True):
    Aur_split = Aur.split(deepcopy=deepcopy)

    print_function_info(f"Component 0, deepcopy={deepcopy}, before", Aur_split[0])
    print_function_info(f"Component 1, deepcopy={deepcopy}, before", Aur_split[1])

    Aur_out = XDMFFile(f"test-0-{deepcopy}.xdmf")
    Aur_out.write_checkpoint(Aur_split[0], "component-0", 0, XDMFFile.Encoding.HDF5, False)
    Aur_out.close()
    Aur_out = XDMFFile(f"test-1-{deepcopy}.xdmf")
    Aur_out.write_checkpoint(Aur_split[1], "component-1", 0, XDMFFile.Encoding.HDF5, False)
    Aur_out.close()

    #Reading in the .xdmf file
    A = Function(VFnSp)
    A_in =  XDMFFile(f"test-0-{deepcopy}.xdmf")
    A_in.read_checkpoint(A,"component-0",0)
    u = Function(VFnSp)
    u_in =  XDMFFile(f"test-1-{deepcopy}.xdmf")
    u_in.read_checkpoint(u,"component-1",0)
    
    #plot(u)
    #plt.title(r"$u(x)-after$",fontsize=26)
    #plt.show()
    #plot(A)
    #plt.title(r"$A(x)e_2-after$",fontsize=26)
    #plt.show()
    
    print_function_info(f"Component 1, deepcopy={deepcopy}, after", A)
    print_function_info(f"Component 1, deepcopy={deepcopy}, fater", u)
