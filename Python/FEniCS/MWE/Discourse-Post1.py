from dolfin import *
import fenics as fe
import matplotlib.pyplot as plt

#Create mesh and define function space
Lx=10.  #length of domain
mesh = fe.IntervalMesh(1000,0,Lx)
x = SpatialCoordinate(mesh)
VA = FiniteElement("CG", mesh.ufl_cell(), 2) #Element for A
Vu = FiniteElement("CG", mesh.ufl_cell(), 2) #Element for u
R = FiniteElement("Real", mesh.ufl_cell(), 0) #Element for Lagrange multiplier
V = FunctionSpace(mesh, MixedElement(VA, Vu, R)) #Creating functionSpace
VFnSp = FunctionSpace(mesh, "Lagrange", 2)
RFnSp = FunctionSpace(mesh, "Real", 0)

#Newton Rhapson input
Aur = interpolate( Expression(("1","0.0", "13.5"), degree=2), V)#SC phase as initial cond.

Aur_split = Aur.split(True)

##Save solution in a .xdmf filedd
#for (component_index, component) in enumerate(Aur_split[:2]):
#    Aur_out = XDMFFile("test-{component_index}.xdmf")
#    Aur_out.write_checkpoint(component, "component-{component_index}", 0, XDMFFile.Encoding.HDF5, False)
#    Aur_out.close()
#with open("test-2.txt", "w") as file:
#    print(float(Aur_split[2]), file=file)


#(A, u, r) = split(Aur)
plot(Aur_split[1])
plt.title(r"$u(x)-b4$",fontsize=26)
plt.show()
plot(Aur_split[0])
plt.title(r"$A(x)e_2-b4$",fontsize=26)
plt.show()


Aur_out = XDMFFile(f"test-0.xdmf")
Aur_out.write_checkpoint(Aur_split[0], "component-0", 0, XDMFFile.Encoding.HDF5, False)
Aur_out.close()
Aur_out = XDMFFile(f"test-1.xdmf")
Aur_out.write_checkpoint(Aur_split[1], "component-1", 0, XDMFFile.Encoding.HDF5, False)
Aur_out.close()
with open(f"test-2.txt", "w") as file:
    print(float(Aur_split[2]), file=file)



#Reading in the .xdmf file 
A = Function(VFnSp)
A_in =  XDMFFile("test-0.xdmf")
A_in.read_checkpoint(A,"component-0",0)
u = Function(VFnSp)
u_in =  XDMFFile("test-1.xdmf")
u_in.read_checkpoint(u,"component-1",0)
r = Function(RFnSp)
r = interpolate(Constant("1.0"), RFnSp)

Aur2 = Function(V)

assign(Aur2, [A,u,r])


plot(u)
plt.title(r"$u(x)-after$",fontsize=26)
plt.show()
plot(A)
plt.title(r"$A(x)e_2-after$",fontsize=26)
plt.show()




