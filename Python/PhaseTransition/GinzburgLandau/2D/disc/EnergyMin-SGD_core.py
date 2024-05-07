#Explicitly model discontinuities while allowing to refine the core seperately.
#======================================================================================================
##Things to do:-
#1. "DONE" Refine the mesh.
#2. read in/store the value of Theta from the last computation (input if its first iteration).
#3. "DONE" Determine \theta1 from \theta.
#4. identify the discontinuity line from the read in value of theta and theta1.
#5. "DONE" Modify the solution of Ft, Fu, Fa1 and Fa2.
#6. When storing the output in an xdmf file, make sure to store theta_1 as well.
#7. Whenever the value of something on the discontinuity crosses 2\pi, we retore it to 2\pi.
#8. "DONE" Implement the different stochastic descent algorithms.
#======================================================================================================
#ISSUES WITH THE CODE:-

import time # timing for performance test.
time0 = time.time()

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

## Defining the Parameters
print("================input to code========================")
pord = int(1)# degree of polynmomials used for FEA
kappa = Constant(2.0)
#lx = float(input("lx? --> "))
#ly = float(input("ly? --> "))
lx = float(10.0)
ly = float(10.0)
print("Code replaces odd no. with next highest even no.")
Nx = int(10*lx/float(kappa)) #int(input("Nx? --> "))
Ny = int(10*ly/float(kappa)) #int(input("Ny? --> "))
Nx = int(2*np.ceil(Nx/2))
Ny = int(2*np.ceil(Ny/2))
gamma = float(0.01) # Learning rate.
NN = int(input('Number of iterations? -->')) # Number of iterations
H = Constant(0.23);
tol = float(0.000001)
read_in = int(1) # 1 to red in o/p of last iteration, 0 for initial condiiton provided here.
c_r = float(kappa)
Ref_No = int(1)
#Ref_No = int(input("Refinement number? -->"))
random_no_method = int(input("0 for Gradient Descent, 1 for Random Stochastic Gradient Descent, 2 for Nesterov, 3 for normal momentum --> "))
tau = float(input("tau? --> ")) # stochastic parameter
if tau < 0 or tau > 1:
  sys.exit("tau cannot be negative")


#Create mesh and define function space
#Nx = max(np.ceil(lx*50/kappa),Nx)
#Ny = max(np.ceil(ly*50/kappa),Ny)
mesh = RectangleMesh(Point(0., 0.), Point(lx, ly), Nx, Ny) # "crossed means that first it will partition the ddomain into Nx x Ny rectangles. Then each rectangle is divided into 4 triangles forming a cross"


plot(mesh)
plt.show()

##Refining the mesh near the centre.
#=================================================================================================================
#!!xDx!! To refine the mesh further, we need to run the following section of code multiple times. USe for loop.

for i in range(Ref_No):
 # Mark cells for refinement
 cell_markers = MeshFunction("bool", mesh, mesh.topology().dim()) # 1st arg is i/p type, 2nd is mesh, 3rd is dimension.
 for c in cells(mesh):
     if c.midpoint().distance(Point(0.5*lx,0.5*ly)) < c_r: # checking if the midpoint of the cell is close to the point p
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

mesh.init(0,1)

#========================================================================================================================
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
#========================================================================================================================


#========================================================================================================================
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
 T1 = Function(V)
 U = Function(V)
 a1_in =  XDMFFile("GL-2DEnrg-0.xdmf")
 a1_in.read_checkpoint(A1,"a1",0)
 a2_in =  XDMFFile("GL-2DEnrg-1.xdmf")
 a2_in.read_checkpoint(A2,"a2",0)
 t_in =  XDMFFile("GL-2DEnrg-2.xdmf")
 t_in.read_checkpoint(T,"t",0)
 t1_in =  XDMFFile("GL-2DEnrg-3.xdmf")
 t1_in.read_checkpoint(T1,"t1",0)
 u_in =  XDMFFile("GL-2DEnrg-4.xdmf")
 u_in.read_checkpoint(U,"u",0)
 #plot(u)
 #plt.title(r"$u(x)-b4$",fontsize=26)
 #plt.show()
else:
 sys.exit("Not a valid input for read_in.")
#========================================================================================================================

a1_up.vector()[:] = A1.vector()[:]
a2_up.vector()[:] = A2.vector()[:]
t_up.vector()[:] = T.vector()[:]
t1_up.vector()[:] = T1.vector()[:]
u_up.vector()[:] = U.vector()[:]

##========================================================================================================================
##Creating a list to store nodes of discontinuity.
xcoord = mesh.coordinates()
v2d = vertex_to_dof_map(V)
d2v = dof_to_vertex_map(V)
disc_node = []
disc_node1 = []

for i,xx in enumerate(xcoord):
 if xx[0] > 0.5*lx and xx[1] == 0.5*ly: # in the right half and along the x2 = 0.5*ly axis.
  disc_node.append(i)
 if xx[0] < 0.5*lx and xx[1] == 0.5*ly: # in the left half and along the x2 = 0.5*ly axis.
  disc_node1.append(i)

n = V.dim()                                                                      
d = mesh.geometry().dim()                                                        
dof_coordinates = V.tabulate_dof_coordinates().reshape(n,d)                      
dof_coordinates.resize((n, d))                                                  
dof_x = dof_coordinates[:, 0]                                                    
dof_y = dof_coordinates[:, 1]                                                    
disc_T = interpolate( T, V)
disc_T1 = interpolate( T1, V)

print("disc_node = ", disc_node)
print("=======================================================")

#========================================================================================================================
## Determining the nodes near the discontinuity.
disc_above = []
disc_below = []
dof_coordinates1 = V.tabulate_dof_coordinates()

##Marking the nodes near the discontinuity.
print("Marking the nodes near the discontinuity.")
for j,yy in enumerate(dof_coordinates1):
 if yy[1] > 0.5*ly and yy[0] > 0.5*lx and (yy[0]-0.5*lx)**2+(yy[1]-0.5*ly)**2 > c_r**2 and abs(yy[1]-0.5*ly) <= ly/Ny + DOLFIN_EPS: # disc nodes above x_2=0.5*ly and right of the core.
  disc_above.append(j)
 elif yy[1] > 0.5*ly and yy[0] > 0.5*lx and (yy[0]-0.5*lx)**2+(yy[1]-0.5*ly)**2 < c_r**2  and abs(yy[1]-0.5*ly) <= ly/Ny/2**(Ref_No)+DOLFIN_EPS: # discnodes above inside core
  disc_above.append(j)
 elif yy[1] < 0.5*ly and yy[0] > 0.5*lx and (yy[0]-0.5*lx)**2+(yy[1]-0.5*ly)**2 > c_r**2 and abs(yy[1]-0.5*ly) <= ly/Ny + DOLFIN_EPS: # disc nodes below x_2=0.5*ly and right of the core.
  disc_below.append(j)
 elif yy[1] < 0.5*ly and yy[0] > 0.5*lx and (yy[0]-0.5*lx)**2+(yy[1]-0.5*ly)**2 < c_r**2 and abs(yy[1]-0.5*ly) <= ly/Ny/2**(Ref_No)+DOLFIN_EPS: # discnodes above&inside core
  disc_below.append(j)

#========================================================================================================================
for tt in range(NN):
 a1.vector()[:] = a1_up.vector()[:]
 a11.vector()[:] = a11_up.vector()[:]
 a2.vector()[:] = a2_up.vector()[:]
 a21.vector()[:] = a21_up.vector()[:]
 t.vector()[:] = t_up.vector()[:]
 t1.vector()[:] = t1_up.vector()[:]
 u.vector()[:] = u_up.vector()[:]
 u1.vector()[:] = u1_up.vector()[:]

 #Checking where the function crosses 0 and 2\pi.
 t_array = np.asarray(t.vector()[:])
 t_array[np.where(t_array<=0)] = 0
 t_array[np.where(t_array>=2*np.pi)] = 2*np.pi
 t.vector()[:] = t_array

 Fa1_vec = assemble(Fa1)
 Fa11_vec = assemble(Fa11)
 Fa2_vec = assemble(Fa2)
 Fa21_vec = assemble(Fa21)
 Ft_vec = assemble(Ft)
 Ft1_vec = assemble(Ft1)
 Fu_vec = assemble(Fu)
 Fu1_vec = assemble(Fu1)

 ##modifying F_\cdot
 for i in disc_node:
  Fa1_vec[v2d[i]] = Fa11_vec[v2d[i]]
  Fa2_vec[v2d[i]] = Fa21_vec[v2d[i]]
  Fu_vec[v2d[i]] = Fu1_vec[v2d[i]]
  Ft_vec[v2d[i]] = Ft1_vec[v2d[i]]
 for i in disc_above:
  Fa1_vec[i] = Fa11_vec[i]
  Fa2_vec[i] = Fa21_vec[i]
  Fu_vec[i] = Fu1_vec[i]
  Ft_vec[i] = Ft1_vec[i]
 for i in disc_below:
  Fa1_vec[i] = Fa11_vec[i]
  Fa2_vec[i] = Fa21_vec[i]
  Fu_vec[i] = Fu1_vec[i]
  Ft_vec[i] = Ft1_vec[i]
   

 a1_up.vector()[:] = a1.vector()[:] - gamma*Fa1_vec[:]
 a2_up.vector()[:] = a2.vector()[:] - gamma*Fa2_vec[:]
 t_up.vector()[:] = t.vector()[:] - gamma*Ft_vec[:]
 u_up.vector()[:] = u.vector()[:] - gamma*Fu_vec[:]


    
 if tt == 0: # For the first iteration, we generate random vec for momentum.
  ma1_vec = [0.0 for _ in range(len(Fa1_vec))] 
  ma2_vec = [0.0 for _ in range(len(Fa2_vec))] 
  mt_vec = [0.0 for _ in range(len(Ft_vec))] 
  mu_vec = [0.0 for _ in range(len(Fu_vec))]
 if random_no_method == 0: # Gradient Descent.
  a1_up.vector()[:] = a1.vector()[:] - gamma*Fa1_vec[:]
  a2_up.vector()[:] = a2.vector()[:] - gamma*Fa2_vec[:]
  t_up.vector()[:] = t.vector()[:] - gamma*Ft_vec[:]
  u_up.vector()[:] = u.vector()[:] - gamma*Fu_vec[:]
 elif random_no_method == 1: # Random Stochastic Gradient Descent.
  ma1_vec = [random.uniform(0,tau) for _ in range(len(Fa1_vec))] # return a random vector of len Fa1_vec each elemnt less than tau. 
  ma2_vec = [random.uniform(0,tau) for _ in range(len(Fa2_vec))] 
  mt_vec = [random.uniform(0,tau) for _ in range(len(Ft_vec))] 
  mu_vec = [random.uniform(0,tau) for _ in range(len(Fu_vec))]
  a1_up.vector()[:] = a1.vector()[:] - gamma*(Fa1_vec[:] + ma1_vec)
  a2_up.vector()[:] = a2.vector()[:] - gamma*(Fa2_vec[:] + ma2_vec)
  t_up.vector()[:] = t.vector()[:] - gamma*(Ft_vec[:] + mt_vec)
  u_up.vector()[:] = u.vector()[:] - gamma*(Fu_vec[:] + mu_vec)
 elif random_no_method == 3: #normal momentum,
  #print("tt=",tt," type of vec =",type(ma1_vec))
  ma1_vec = [ii * tau for ii in ma1_vec]
  ma1_vec[:] = ma1_vec[:] - gamma*Fa1_vec[:] 
  a1_up.vector()[:] = a1.vector()[:] + ma1_vec
  ma2_vec = [ii * tau for ii in ma2_vec]
  ma2_vec[:] = ma2_vec[:] - gamma*Fa2_vec[:] 
  a2_up.vector()[:] = a2.vector()[:] + ma2_vec
  mt_vec = [ii * tau for ii in mt_vec]
  mt_vec[:] = mt_vec[:] - gamma*Ft_vec[:] 
  t_up.vector()[:] = t.vector()[:] + mt_vec
  mu_vec = [ii * tau for ii in mu_vec]
  mu_vec[:] = mu_vec[:] - gamma*Fu_vec[:] 
  u_up.vector()[:] = u.vector()[:] + mu_vec
 elif random_no_method == 2: #Nesterov.
   sys.exit("Not ready yet.")
 else:
  sys.exit("Not a valid input for random_no_method.")

 # Obtaining t1 from t.
 t1_array = t1_up.vector()[:]
 t1_array = t_up.vector()[:]
 for i in range(0,len(t1_array)):
  if 0 <= t1_array[i] <= np.pi:
   t1_array[i] = t1_array[i] + 2*np.pi
 t1_up.vector()[:] = t1_array

 
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
a1a2tt1u_out = XDMFFile('GL-2DEnrg-0.xdmf')
a1a2tt1u_out.write_checkpoint(a1, "a1", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("GL-2DEnrg-0.pvd") # for paraview.
pvd_file << a1
a1a2tt1u_out.close()
a1a2tt1u_out = XDMFFile('GL-2DEnrg-1.xdmf')
a1a2tt1u_out.write_checkpoint(a2, "a2", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("GL-2DEnrg-1.pvd") # for paraview.
pvd_file << a2
a1a2tt1u_out.close()
a1a2tt1u_out = XDMFFile('GL-2DEnrg-2.xdmf')
a1a2tt1u_out.write_checkpoint(t, "t", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("GL-2DEnrg-2.pvd") # for paraview.
pvd_file << t
a1a2tt1u_out.close()
a1a2tt1u_out = XDMFFile('GL-2DEnrg-3.xdmf')
a1a2tt1u_out.write_checkpoint(u, "t1", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("GL-2DEnrg-3.pvd")
pvd_file << t1
a1a2tt1u_out.close()
a1a2tt1u_out = XDMFFile('GL-2DEnrg-4.xdmf')
a1a2tt1u_out.write_checkpoint(u, "u", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("GL-2DEnrg-4.pvd")
pvd_file << u
a1a2tt1u_out.close()


pie = assemble((1/(lx*ly))*((1-u**2)**2/2 + (1/kappa**2)*inner(grad(u), grad(u)) \
                        + ( (a1-t.dx(0))**2 + (a2-t.dx(1))**2 )*u**2 \
                            + inner( curl(a1 ,a2-Ae), curl(a1 ,a2-Ae) ) )*dx )


print("================output of code========================")
print("Energy density is", pie)
print("gamma = ", gamma)
print("kappa = ", kappa)
print("lx = ", lx)
print("ly = ", ly)
print("Nx = ", Nx)
print("Ny = ", Ny)
print("NN = ", NN)
print("H = ", float(H))
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

time1 = time.time()

print("time taken for code to run = ", time1-time0)
