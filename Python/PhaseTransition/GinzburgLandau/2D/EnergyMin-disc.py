#Explicitly model disocntinuities.
#======================================================================================================
#1. identify the nodes associated with the discontinuity.
#   a. Check where T is discontinuous,
#   b. Chcek to make sure that u, A and t for these variables develop a discontinuity.
#2. try to modify the values of the gradients at that point.  
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
#mesh = RectangleMesh(Point(0., 0.), Point(lx, ly), 1+np.ceil(lx*10/kappa), 1+np.ceil(ly*10/kappa)) # "crossed means that first it will partition the ddomain into Nx x Ny rectangles. Then each rectangle is divided into 4 triangles forming a cross"
mesh = RectangleMesh(Point(0., 0.), Point(lx, ly), 7, 7) # "crossed means that first it will partition the ddomain into Nx x Ny rectangles. Then each rectangle is divided into 4 triangles forming a cross"
x = SpatialCoordinate(mesh)
Ae = H*x[0] #The vec pot is A(x) = Hx_1e_2
V = FunctionSpace(mesh, "Lagrange", 2)#This is for ExtFile
Vt = FunctionSpace(mesh, "DG", 2)#This is for ExtFile

mesh.init(0,1)

#for v in vertices(mesh):
#    idx = v.index()
#    neighborhood = [Edge(mesh, i).entities(0) for i in v.entities(1)]
#    neighborhood = np.array(neighborhood).flatten()
#
#    # Remove own index from neighborhood
#    neighborhood = neighborhood[np.where(neighborhood != idx)[0]]
#    print("Vertex %d neighborhood: " %idx, neighborhood)


# Define functions
a1 = Function(V)
a2 = Function(V)
t = Function(Vt)
t1 = Function(Vt)
u = Function(V)
a1_up = Function(V)
a2_up = Function(V)
t_up = Function(Vt)
t1_up = Function(Vt)
u_up = Function(V)
#Temp functions to store the frechet derivatives
temp_a1 = Function(V)
temp_a2 = Function(V)
temp_t = Function(Vt)
temp_t1 = Function(Vt)
temp_u = Function(V)

def curl(a1,a2):
    return a2.dx(0) - a1.dx(1)

#Defining the energy
Pi = ( (1-u**2)**2/2 + (1/kappa**2)*inner(grad(u), grad(u)) + ( (a1-t.dx(0))**2 + (a2-t.dx(1))**2 )*u**2 \
      + inner( curl(a1 ,a2-Ae), curl(a1 ,a2-Ae) ) )*dx
Pi1 = ( (1-u**2)**2/2 + (1/kappa**2)*inner(grad(u), grad(u)) + ( (a1-t1.dx(0))**2 + (a2-t1.dx(1))**2 )*u**2 \
      + inner( curl(a1 ,a2-Ae), curl(a1 ,a2-Ae) ) )*dx

#Defining the gradient
Fa1 = derivative(Pi, a1)
Fa2 = derivative(Pi, a2)
Ft = derivative(Pi, t)
Ft1 = derivative(Pi1, t1)
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
# T = interpolate( Expression('(x[0]-0.5*lx)*(x[0]-0.5*lx)+(x[1]-0.5*ly)*(x[1]-0.5*ly) <= r*r + DOLFIN_EPS ? 1 \
#                             : atan2(x[0]-0.5*lx,x[1]-0.5*ly)', lx=lx, ly=ly, r=0.001, degree=2), Vt)
# T1 = interpolate( Expression('(x[0]-0.5*lx)*(x[0]-0.5*lx)+(x[1]-0.5*ly)*(x[1]-0.5*ly) <= r*r + DOLFIN_EPS ? 1 \
#                             : -0.5*pie+atan2(x[1]-0.5*ly,-x[0]+0.5*lx)',pie=np.pi, lx=lx, ly=ly, r=0.001, degree=2), Vt) # 0.5*np.pi+
 T = interpolate( Expression('atan2(x[0]-0.5*lx,x[1]-0.5*ly)', lx=lx, ly=ly, r=0.001, degree=2), Vt)
 T1 = interpolate( Expression('-0.5*pie+atan2(x[1]-0.5*ly,-x[0]+0.5*lx)',pie=np.pi, lx=lx, ly=ly, r=0.001, degree=2), Vt) # 0.5*np.pi+
 U = interpolate( Expression('tanh(sqrt((x[0]-0.5*lx)*(x[0]-0.5*lx)+(x[1]-0.5*ly)*(x[1]-0.5*ly)))', lx=lx, ly=ly, degree=2), V) 
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
t1_up.vector()[:] = T1.vector()[:]
u_up.vector()[:] = U.vector()[:]

##identifying the branch cut
#check where T is dicontinuous.
list = [];
for v in vertices(mesh):
    #print("in the vertex loop")
    idx = v.index()
    neighborhood = [Edge(mesh, i).entities(0) for i in v.entities(1)]
    neighborhood = np.array(neighborhood).flatten()

    # Remove own index from neighborhood
    neighborhood = neighborhood[np.where(neighborhood != idx)[0]]
    for ii in neighborhood:
     print("|T(",ii,")-T(",idx,")|=",np.absolute(T.vector()[ii]-T.vector()[idx]))
     if np.absolute(T.vector()[ii] - T.vector()[idx]) >= 2*np.pi-0.2 :
      print("inside if")
      list.append([ii,idx])
      print("T(",ii,")=",T.vector()[ii],"and T(",idx,")=",T.vector()[idx])
print(list)


#Plot mesh
plot(mesh)
#plot(T.vector())
plt.show()

coordinates = mesh.coordinates()
print(coordinates)

#mesh_Test = interpolate( Expression('x[0]+x[1]', degree=1), V)
#plot(


for tt in range(NN):
 a1.vector()[:] = a1_up.vector()[:]
 a2.vector()[:] = a2_up.vector()[:]
 t.vector()[:] = t_up.vector()[:] 
 t1.vector()[:] = t1_up.vector()[:] 
 #if any(t.vector()[:] < 0) or any(t.vector()[:] > 2*np.pi):
 # print("============================================================")
 # print("before modding the previous output")
 # print(t_up.vector()[:])
 # print("============================================================")
 # print("after modding the previous output")
 # print(t.vector()[:])
 u.vector()[:] = u_up.vector()[:]
 Fa1_vec = assemble(Fa1)
 Fa2_vec = assemble(Fa2)
 Ft_vec = assemble(Ft)
 Ft1_vec = assemble(Ft1)
 Fu_vec = assemble(Fu)
 
 #modifying F_t
 #for v in vertices(mesh):
 #   #print("in the vertex loop")
 #   idx = v.index()
 #   neighborhood = [Edge(mesh, i).entities(0) for i in v.entities(1)]
 #   neighborhood = np.array(neighborhood).flatten()

 #   # Remove own index from neighborhood
 #   neighborhood = neighborhood[np.where(neighborhood != idx)[0]]
 #   for ii in neighborhood:
 #    if np.absolute(t.vector()[ii] - t.vector()[idx]) >= 2*np.pi-0.001 :
 #     print("t(",ii,")=",t.vector()[ii],"and t(",idx,")=",t.vector()[idx])
 #    

 a1_up.vector()[:] = a1.vector()[:] - gamma*Fa1_vec[:]
 a2_up.vector()[:] = a2.vector()[:] - gamma*Fa2_vec[:]
 t_up.vector()[:] = t.vector()[:] - gamma*Ft_vec[:]
 t1_up.vector()[:] = t1.vector()[:] - gamma*Ft1_vec[:]
 u_up.vector()[:] = u.vector()[:] - gamma*Fu_vec[:]
 temp_a1.vector()[:] = Fa1_vec[:]
 temp_a2.vector()[:] = Fa2_vec[:]
 temp_t.vector()[:] = Ft_vec[:]
 temp_t1.vector()[:] = Ft1_vec[:]
 temp_u.vector()[:] = Fu_vec[:]
 
 #c = plot(temp_u)
 #plt.title(r"$F_{u}(x)$",fontsize=26)
 #plt.colorbar(c)
 #plt.show()
 #c = plot(temp_a1)
 #plt.title(r"$F_{a1}(x)$",fontsize=26)
 #plt.colorbar(c)
 #plt.show()
 #c = plot(temp_a2)
 #plt.title(r"$F_{a2}(x)$",fontsize=26)
 #plt.colorbar(c)
 #plt.show()
 #c = plot(temp_t)
 #plt.title(r"$F_{\theta}$(x)",fontsize=26)
 #plt.colorbar(c)
 #plt.show()
 #c = plot(temp_t1)
 #plt.title(r"$F_{\theta1}$(x)",fontsize=26)
 #plt.colorbar(c)
 #plt.show()

 #print(Fa1_vec.get_local()) # prints the vector.
 #print(np.linalg.norm(np.asarray(Fa1_vec.get_local()))) # prints the vector's norm.
 tol_test = np.linalg.norm(np.asarray(Fa1_vec.get_local()))\
           +np.linalg.norm(np.asarray(Fa2_vec.get_local()))\
           +np.linalg.norm(np.asarray(Ft_vec.get_local()))\
           +np.linalg.norm(np.asarray(Fu_vec.get_local()))
 print(tol_test)
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

c = plot(U)
plt.title(r"$U(x)$",fontsize=26)
plt.colorbar(c)
plt.show()
c = plot(T)
plt.title(r"$T(x)$",fontsize=26)
plt.colorbar(c)
plt.show()
c = plot(T1)
plt.title(r"$T1(x)$",fontsize=26)
plt.colorbar(c)
plt.show()

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
