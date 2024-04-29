#Explicitly model disocntinuities, but we mesh near the core finer.
#======================================================================================================
#1. identify the nodes associated with the discontinuity.
# "DONE"  a. Check where T is discontinuous,  
#   b. Check to make sure that u, A and t for these variables develop a discontinuity.
# "DONE" 2. print the values of the nodes and check which ones are closest to the discontinuity.  
# "DONE" 3. Try to modify the values of the gradients at that point. 
# "DONE"   print \nabla \theta  and see if you have a discontinuity there. 
# "DONE" 4. Get N_x, N_y from kappa. Implenet the 2*np.ceil((1+N)/2)
#5. Properly modify theta, theta1 such that it is bounded within the appropriate range. 
#6. Plug in the closed form solution near the origin. Maybe use two domains. with different meshing.
#IMP. plot the discontinuity for \theta
#IMP. Make sure that its along x1>0, x2=0.
#IMP. Get \theta_1 from rotating and shifting \theta.
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

import sys
np.set_printoptions(threshold=sys.maxsize)

#Parameters
print("================input to code========================")
pord = int(1)# degree of polynmomials used for FEA
kappa = Constant(2.0)
lx = float(input("lx? --> "))
ly = float(input("ly? --> "))
print("Code replaces odd no. with next highest even no.")
Nx = int(input("Nx? --> "))
Ny = int(input("Ny? --> "))
Nx = int(2*np.ceil(Nx/2))
Ny = int(2*np.ceil(Ny/2))
gamma = float(0.01) # Learning rate.
NN = int(input('Number of iterations? -->')) # Number of iterations
H = Constant(0.23);
tol = float(0.000001)
read_in = int(0)
c_r = float(np.sqrt(0.1))
Ref_No = int(input("Refinement number? -->"))


#Create mesh and define function space
#Nx = max(np.ceil(lx*10/kappa),Nx)
#Ny = max(np.ceil(ly*10/kappa),Ny)
mesh = RectangleMesh(Point(0., 0.), Point(lx, ly), Nx, Ny) # "crossed means that first it will partition the ddomain into Nx x Ny rectangles. Then each rectangle is divided into 4 triangles forming a cross"


plot(mesh)
plt.show()

##Refining the mesh near the centre.=================================================================================
#!!xDx!! To refine the mesh further, we need to run the following section of code multiple times. USe for loop.

for i in range(Ref_No):
 # Mark cells for refinement
 cell_markers = MeshFunction("bool", mesh, mesh.topology().dim()) # 1st arg is i/p type, 2nd is mesh, 3rd is dimension.
 for c in cells(mesh):
     if c.midpoint().distance(Point(0.5*lx,0.5*ly)) < c_r**2: # checking if the midpoint of the cell is close to the point p
         cell_markers[c] = True
     else:
         cell_markers[c] = False
 mesh = refine(mesh, cell_markers)
 plot(mesh)
 plt.show()
##====================================================================================================================

x = SpatialCoordinate(mesh)
Ae = H*x[0] #The vec pot is A(x) = Hx_1e_2
V = FunctionSpace(mesh, "Lagrange", pord)#This is for ExtFile
Vt = FunctionSpace(mesh, "DG", pord)#This is for ExtFile

mesh.init(0,1)

# Define functions
a1 = Function(V)
a11 = Function(V)
a2 = Function(V)
a21 = Function(V)
t = Function(V)
t1 = Function(V)
u = Function(V)
u1 = Function(V)
a1_up = Function(V)
a11_up = Function(V)
a2_up = Function(V)
a21_up = Function(V)
t_up = Function(V)
t1_up = Function(V)
u_up = Function(V)
u1_up = Function(V)
#Temp functions to store the frechet derivatives
temp_a1 = Function(V)
temp_a11 = Function(V)
temp_a2 = Function(V)
temp_a21 = Function(V)
temp_t = Function(V)
temp_t1 = Function(V)
temp_u = Function(V)
temp_u1 = Function(V)

def curl(a1,a2):
    return a2.dx(0) - a1.dx(1)

#Defining the energy
Pi = ( (1-u**2)**2/2 + (1/kappa**2)*inner(grad(u), grad(u)) + ( (a1-t.dx(0))**2 + (a2-t.dx(1))**2 )*u**2 \
      + inner( curl(a1 ,a2-Ae), curl(a1 ,a2-Ae) ) )*dx
Pi1 = ( (1-u**2)**2/2 + (1/kappa**2)*inner(grad(u), grad(u)) + ( (a1-t1.dx(0))**2 + (a2-t1.dx(1))**2 )*u**2 \
      + inner( curl(a1 ,a2-Ae), curl(a1 ,a2-Ae) ) )*dx

#Defining the gradients for each branch of the Riemann manifold.
Fa1 = derivative(Pi, a1)
Fa11 = derivative(Pi1, a1)
Fa2 = derivative(Pi, a2)
Fa21 = derivative(Pi1, a2)
Ft = derivative(Pi, t)
Ft1 = derivative(Pi1, t1)
Fu = derivative(Pi, u)
Fu1 = derivative(Pi1, u)


##Setting up the initial conditions
if read_in == 0: # We want to use the standard values.
 ##SC state
 #print("Using bulk SC as initial condition")
 #A1 = interpolate( Expression("0.0", degree=pord), V)
 #A2 = interpolate( Expression("0.0", degree=pord), V)
 #T = interpolate( Expression("1.0", degree=pord), V)
 #U = interpolate( Expression("1.0", degree=pord), V)
 ##Modified normal state
 #print("Using modified bulk Normal as initial condition")
 #A1 = interpolate( Expression("0.0", degree=pord), V)
 #A2 = interpolate( Expression("H*x[0]", H=H, degree=pord), V)
 #T = interpolate( Expression("x[1]", degree=pord), V)
 #U = interpolate( Expression("x[0]", degree=pord), V)
 ##Vortex Solution.
 print("Using Vortex solution")
 A1 = interpolate( Expression('sqrt((x[0]-0.5*lx)*(x[0]-0.5*lx)+(x[1]-0.5*ly)*(x[1]-0.5*ly)) <= r + DOLFIN_EPS ? -x[1] : \
                             -exp(-sqrt((x[0]-0.5*lx)*(x[0]-0.5*lx)+(x[1]-0.5*ly)*(x[1]-0.5*ly))) \
                              *x[1]/sqrt((x[0]-0.5*lx)*(x[0]-0.5*lx)+(x[1]-0.5*ly)*(x[1]-0.5*ly))*1/K', \
                                lx=lx, ly=ly, r=0.3517, K=kappa, degree=pord), V)
 A2 = interpolate( Expression('sqrt((x[0]-0.5*lx)*(x[0]-0.5*lx)+(x[1]-0.5*ly)*(x[1]-0.5*ly)) <= r + DOLFIN_EPS ? x[0] : \
                             exp(-sqrt((x[0]-0.5*lx)*(x[0]-0.5*lx)+(x[1]-0.5*ly)*(x[1]-0.5*ly))) \
                              *x[0]/sqrt((x[0]-0.5*lx)*(x[0]-0.5*lx)+(x[1]-0.5*ly)*(x[1]-0.5*ly))*1/K', \
                                lx=lx, ly=ly, r=0.3517, K=kappa, degree=pord), V)
# T = interpolate( Expression('(x[0]-0.5*lx)*(x[0]-0.5*lx)+(x[1]-0.5*ly)*(x[1]-0.5*ly) <= r*r + DOLFIN_EPS ? 1 \
#                             : atan2(x[0]-0.5*lx,x[1]-0.5*ly)', lx=lx, ly=ly, r=0.001, degree=pord), V)
# T1 = interpolate( Expression('(x[0]-0.5*lx)*(x[0]-0.5*lx)+(x[1]-0.5*ly)*(x[1]-0.5*ly) <= r*r + DOLFIN_EPS ? 1 \
#                             : pie+atan2(x[1]-0.5*ly,-x[0]+0.5*lx)',pie=np.pi, lx=lx, ly=ly, r=0.001, degree=pord), V) # 0.5*np.pi+
#### !!xDx!! atan2(f1,f2) = atan(f1/f2)
 T = interpolate( Expression('atan2(-x[1]+0.5*ly,-x[0]+0.5*lx)+pie',pie=np.pi, lx=lx, ly=ly, degree=pord), V)
 T1 = interpolate( Expression('atan2(x[1]-0.5*lx,x[0]-0.5*ly)+2*pie',pie=np.pi, lx=lx, ly=ly, degree=pord), V) 
 U = interpolate( Expression('tanh(sqrt((x[0]-0.5*lx)*(x[0]-0.5*lx)+(x[1]-0.5*ly)*(x[1]-0.5*ly)))', lx=lx, ly=ly, degree=pord), V) 
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
 sys.exit("Not a valid input for read_in.")

a1_up.vector()[:] = A1.vector()[:]
a2_up.vector()[:] = A2.vector()[:]
t_up.vector()[:] = T.vector()[:]
t1_up.vector()[:] = T1.vector()[:]
u_up.vector()[:] = U.vector()[:]

#Plot mesh
plot(mesh)
plt.show()

#Creating a vector to mark the discontinuity.
Marker = interpolate( Expression('0', degree=pord), V)
Marker1 = interpolate( Expression('3*pie', pie=np.pi, degree=pord), V)

Marker_array = Marker.vector()[:]
Marker_array1 = Marker1.vector()[:]
xcoord = mesh.coordinates()
v2d = vertex_to_dof_map(V)
d2v = dof_to_vertex_map(V)
disc_node = []
disc_node1 = []

for i,xx in enumerate(xcoord):
 if xx[0] >= 0.5*lx and xx[1] == 0.5*ly:
  #print("on the discontinuity",xx)
  #print("index no. is ",i)
  Marker_array[v2d[i]] = np.pi 
  disc_node.append(i)
 if xx[0] <= 0.5*lx and xx[1] == 0.5*ly:
  #print("on the discontinuity1",xx)
  #print("index1 no. is ",i)
  Marker_array1[v2d[i]] = np.pi 
  disc_node1.append(i)

Marker.vector()[:] = Marker_array
Marker1.vector()[:] = Marker_array1

n = V.dim()                                                                      
d = mesh.geometry().dim()                                                        
dof_coordinates = V.tabulate_dof_coordinates().reshape(n,d)                      
dof_coordinates.resize((n, d))                                                   
dof_x = dof_coordinates[:, 0]                                                    
dof_y = dof_coordinates[:, 1]                                                    
disc_T = interpolate( T, V)
disc_T1 = interpolate( T1, V)


##Plotting the functions around the discontinuity to check 
print("Testing for the discontinuity in the variables.")
#Need to identify the nodes nearest to the discontinuity.
print("------------------------------------------------------------------------")



## Scatter plot to map discontinuity.
fig = plt.figure()                                                               
ax = fig.add_subplot(111, projection='3d')                                       
ax.scatter(dof_x, dof_y, disc_T.vector()[:], c='g', marker='.')                  
ax.scatter(dof_x, dof_y, Marker.vector()[:], c='r', marker='.')                  
#ax.scatter(dof_x, dof_y, Marker1.vector()[:], c='m', marker='.')                  
plt.xlabel("x")
plt.ylabel("y")
plt.show()                                                                       

## Scatter plot to map discontinuity.
fig = plt.figure()                                                               
ax = fig.add_subplot(111, projection='3d')                                       
ax.scatter(dof_x, dof_y, disc_T1.vector()[:], c='b', marker='.')                  
#ax.scatter(dof_x, dof_y, Marker.vector()[:], c='r', marker='.')                  
ax.scatter(dof_x, dof_y, Marker1.vector()[:], c='m', marker='.')                  
plt.xlabel("x")
plt.ylabel("y")
plt.show()                                                                       

for tt in range(NN):
 print("============================================================")
 print("iteration number = ",tt)
 print("============================================================")
 a1.vector()[:] = a1_up.vector()[:]
 a11.vector()[:] = a11_up.vector()[:]
 a2.vector()[:] = a2_up.vector()[:]
 a21.vector()[:] = a21_up.vector()[:]
 t.vector()[:] = t_up.vector()[:] 
 t1.vector()[:] = t1_up.vector()[:] 
 if any(t.vector()[:] < 0) or any(t.vector()[:] > 2*np.pi):
  print("============================================================")
  print("before modding the previous output")
  print(t_up.vector()[:])
  print("============================================================")
  print("after modding the previous output")
  print(t.vector()[:])
 u.vector()[:] = u_up.vector()[:]
 u1.vector()[:] = u1_up.vector()[:]
 Fa1_vec = assemble(Fa1)
 Fa11_vec = assemble(Fa11)
 Fa2_vec = assemble(Fa2)
 Fa21_vec = assemble(Fa21)
 Ft_vec = assemble(Ft)
 Ft1_vec = assemble(Ft1)
 Fu_vec = assemble(Fu)
 Fu1_vec = assemble(Fu1)

 ###Plotting the Reiz representation of the Freschet derivatives.
 ### based on tests here, it seems best to replace i-1, i and i+1 with the their corresponding branch 1 values.
 #store_t = np.zeros(7)
 #store_u = np.zeros(7)
 #store_a1 = np.zeros(7)
 #store_a2 = np.zeros(7)
 #for i in vertex_coord:
 # store_t[0] = store_t[0] + np.absolute(Ft_vec[v2d[i-3]] - Ft1_vec[v2d[i-3]])
 # store_t[1] = store_t[1] + np.absolute(Ft_vec[v2d[i-2]] - Ft1_vec[v2d[i-2]])
 # store_t[2] = store_t[2] + np.absolute(Ft_vec[v2d[i-1]] - Ft1_vec[v2d[i-1]])
 # store_t[3] = store_t[3] + np.absolute(Ft_vec[v2d[i-0]] - Ft1_vec[v2d[i-0]])
 # store_t[4] = store_t[4] + np.absolute(Ft_vec[v2d[i+1]] - Ft1_vec[v2d[i+1]])
 # store_t[5] = store_t[5] + np.absolute(Ft_vec[v2d[i+2]] - Ft1_vec[v2d[i+2]])
 # store_t[6] = store_t[6] + np.absolute(Ft_vec[v2d[i+3]] - Ft1_vec[v2d[i+3]])

 # store_u[0] = store_u[0] + np.absolute(Fu_vec[v2d[i-3]] - Fu1_vec[v2d[i-3]])
 # store_u[1] = store_u[1] + np.absolute(Fu_vec[v2d[i-2]] - Fu1_vec[v2d[i-2]])
 # store_u[2] = store_u[2] + np.absolute(Fu_vec[v2d[i-1]] - Fu1_vec[v2d[i-1]])
 # store_u[3] = store_u[3] + np.absolute(Fu_vec[v2d[i-0]] - Fu1_vec[v2d[i-0]])
 # store_u[4] = store_u[4] + np.absolute(Fu_vec[v2d[i+1]] - Fu1_vec[v2d[i+1]])
 # store_u[5] = store_u[5] + np.absolute(Fu_vec[v2d[i+2]] - Fu1_vec[v2d[i+2]])
 # store_u[6] = store_u[6] + np.absolute(Fu_vec[v2d[i+3]] - Fu1_vec[v2d[i+3]])

 # store_a1[0] = store_a1[0] + np.absolute(Fa1_vec[v2d[i-3]] - Fa11_vec[v2d[i-3]])
 # store_a1[1] = store_a1[1] + np.absolute(Fa1_vec[v2d[i-2]] - Fa11_vec[v2d[i-2]])
 # store_a1[2] = store_a1[2] + np.absolute(Fa1_vec[v2d[i-1]] - Fa11_vec[v2d[i-1]])
 # store_a1[3] = store_a1[3] + np.absolute(Fa1_vec[v2d[i-0]] - Fa11_vec[v2d[i-0]])
 # store_a1[4] = store_a1[4] + np.absolute(Fa1_vec[v2d[i+1]] - Fa11_vec[v2d[i+1]])
 # store_a1[5] = store_a1[5] + np.absolute(Fa1_vec[v2d[i+2]] - Fa11_vec[v2d[i+2]])
 # store_a1[6] = store_a1[6] + np.absolute(Fa1_vec[v2d[i+3]] - Fa11_vec[v2d[i+3]])

 # store_a2[0] = store_a2[0] + np.absolute(Fa2_vec[v2d[i-3]] - Fa21_vec[v2d[i-3]])
 # store_a2[1] = store_a2[1] + np.absolute(Fa2_vec[v2d[i-2]] - Fa21_vec[v2d[i-2]])
 # store_a2[2] = store_a2[2] + np.absolute(Fa2_vec[v2d[i-1]] - Fa21_vec[v2d[i-1]])
 # store_a2[3] = store_a2[3] + np.absolute(Fa2_vec[v2d[i-0]] - Fa21_vec[v2d[i-0]])
 # store_a2[4] = store_a2[4] + np.absolute(Fa2_vec[v2d[i+1]] - Fa21_vec[v2d[i+1]])
 # store_a2[5] = store_a2[5] + np.absolute(Fa2_vec[v2d[i+2]] - Fa21_vec[v2d[i+2]])
 # store_a2[6] = store_a2[6] + np.absolute(Fa2_vec[v2d[i+3]] - Fa21_vec[v2d[i+3]])

 #print("iteration = ", tt)
 #print("------------------------------------------------------------------------") 
 #print("sum||Ft_vec-Ft1_vec|_cdot-3||= ", store_t[0]) 
 #print("sum||Ft_vec-Ft1_vec|_cdot-2||= ", store_t[1]) 
 #print("sum||Ft_vec-Ft1_vec|_cdot-1||= ", store_t[2]) 
 #print("sum||Ft_vec-Ft1_vec|_cdot-0||= ", store_t[3]) 
 #print("sum||Ft_vec-Ft1_vec|_cdot+1||= ", store_t[4]) 
 #print("sum||Ft_vec-Ft1_vec|_cdot+2||= ", store_t[5]) 
 #print("sum||Ft_vec-Ft1_vec|_cdot+3||= ", store_t[6]) 
 #print("------------------------------------------------------------------------") 
 #print("sum||Fu_vec-Fu1_vec|_cdot-3||= ", store_u[0]) 
 #print("sum||Fu_vec-Fu1_vec|_cdot-2||= ", store_u[1]) 
 #print("sum||Fu_vec-Fu1_vec|_cdot-1||= ", store_u[2]) 
 #print("sum||Fu_vec-Fu1_vec|_cdot-0||= ", store_u[3]) 
 #print("sum||Fu_vec-Fu1_vec|_cdot+1||= ", store_u[4]) 
 #print("sum||Fu_vec-Fu1_vec|_cdot+2||= ", store_u[5]) 
 #print("sum||Fu_vec-Fu1_vec|_cdot+3||= ", store_u[6]) 
 #print("------------------------------------------------------------------------") 
 #print("sum||Fa1_vec-Fa11_vec|_cdot-3||= ", store_a1[0]) 
 #print("sum||Fa1_vec-Fa11_vec|_cdot-2||= ", store_a1[1]) 
 #print("sum||Fa1_vec-Fa11_vec|_cdot-1||= ", store_a1[2]) 
 #print("sum||Fa1_vec-Fa11_vec|_cdot-0||= ", store_a1[3]) 
 #print("sum||Fa1_vec-Fa11_vec|_cdot+1||= ", store_a1[4]) 
 #print("sum||Fa1_vec-Fa11_vec|_cdot+2||= ", store_a1[5]) 
 #print("sum||Fa1_vec-Fa11_vec|_cdot+3||= ", store_a1[6]) 
 #print("------------------------------------------------------------------------") 
 #print("sum||Fa2_vec-Fa21_vec|_cdot-3||= ", store_a2[0]) 
 #print("sum||Fa2_vec-Fa21_vec|_cdot-2||= ", store_a2[1]) 
 #print("sum||Fa2_vec-Fa21_vec|_cdot-1||= ", store_a2[2]) 
 #print("sum||Fa2_vec-Fa21_vec|_cdot-0||= ", store_a2[3]) 
 #print("sum||Fa2_vec-Fa21_vec|_cdot+1||= ", store_a2[4]) 
 #print("sum||Fa2_vec-Fa21_vec|_cdot+2||= ", store_a2[5]) 
 #print("sum||Fa2_vec-Fa21_vec|_cdot+3||= ", store_a2[6]) 
 #print("------------------------------------------------------------------------") 
  

 ##modifying F_t
 for i in vertex_coord:
  Fa1_vec[v2d[i-1:i+1:1]] = Fa11_vec[v2d[i-1:i+1:1]]
  Fa2_vec[v2d[i-1:i+1:1]] = Fa21_vec[v2d[i-1:i+1:1]]
  Fu_vec[v2d[i-1:i+1:1]] = Fu1_vec[v2d[i-1:i+1:1]]
  Ft_vec[v2d[i-1:i+1:1]] = Ft1_vec[v2d[i-1:i+1:1]]
    

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
 
 #c = plot(temp_t)
 #plt.title(r"$F_{\theta}$(x)",fontsize=26)
 #plt.colorbar(c)
 #plt.show()
 #c = plot(temp_t1)
 #plt.title(r"$F_{\theta1}$(x)",fontsize=26)
 #plt.colorbar(c)
 #plt.show()
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
#print("Energy density is", pie)
#print("gamma = ", gamma)
#print("kappa = ", kappa)
#print("lx = ", lx)
#print("ly = ", ly)
print("Nx = ", Nx)
print("Ny = ", Ny)
#print("NN = ", NN)
#print("H = ", H)
#print("tol = ", tol, ", ", float(tol_test))
#print("read_in = ", read_in)

#c = plot(U)
#plt.title(r"$U(x)$",fontsize=26)
#plt.colorbar(c)
#plt.show()
#c = plot(T)
#plt.title(r"$T(x)$",fontsize=26)
#plt.colorbar(c)
#plt.show()
#c = plot(T1)
#plt.title(r"$T1(x)$",fontsize=26)
#plt.colorbar(c)
#plt.show()

#c = plot(u)
#plt.title(r"$u(x)$",fontsize=26)
#plt.colorbar(c)
#plt.show()
#c = plot(a1)
#plt.title(r"$A_1(x)$",fontsize=26)
#plt.colorbar(c)
#plt.show()
#c = plot(a2)
#plt.title(r"$A_2(x)$",fontsize=26)
#plt.colorbar(c)
#plt.show()
#c = plot(t)
#plt.title(r"$\theta(x)$",fontsize=26)
#plt.colorbar(c)
#plt.show()

t1 = time.time()

print("time taken for code to run = ", t1-t0)

