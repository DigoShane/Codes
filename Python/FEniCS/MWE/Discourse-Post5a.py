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
pord = int(1)# degree of polynmomials used for FEA
lx = float(1.0)
ly = float(1.0)
Nx = int(40)
Ny = int(40)
#c_r = float(0.1)
#Ref_No = int(input("Refinement number? -->"))


#Create mesh and define function space
mesh = RectangleMesh(Point(0., 0.), Point(lx, ly), Nx, Ny) 
x = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, "Lagrange", pord)

##Reading in function
T = interpolate( Expression('atan2(-0.5*(x[0]-0.5*lx)+rt*0.5*(x[1]-0.5*ly), 0.5*(x[1]-0.5*ly)+rt*0.5*(x[0]-0.5*lx) )', rt=np.sqrt(3), lx=lx, ly=ly, degree=pord), V)

###---------------------------------------------------------------------------------------------------------------
Tarray = T.vector()[:]

##====================================================================================================================
#Section of code to visualize the discontinuity.
Marker = interpolate( Expression('-pie', pie=np.pi, degree=pord), V)
Marker1 = interpolate( Expression('-pie', pie=np.pi, degree=pord), V)

Marker_array = Marker.vector()[:]
Marker_array1 = Marker1.vector()[:]
d2v = dof_to_vertex_map(V)

for i,t in enumerate(Tarray):
 if t == -np.pi + DOLFIN_EPS:
  Marker_array[d2v[i]] = 2*np.pi
 if t == np.pi - DOLFIN_EPS:
  Marker_array1[d2v[i]] = 2*np.pi

print("location of 0's", np.where( np.array(Tarray) == 0))
print("location of 2*pi's", np.where( np.array(Tarray) == 2*np.pi))

Marker.vector()[:] = Marker_array
Marker1.vector()[:] = Marker_array1

T.vector()[:] = Tarray

c = plot(T)
plt.title(r"$\theta$(x)",fontsize=26)
plt.colorbar(c)
plt.show()


n = V.dim()                                                                      
d = mesh.geometry().dim()                                                        
dof_coordinates = V.tabulate_dof_coordinates().reshape(n,d)                      
dof_coordinates.resize((n, d))                                                   
dof_x = dof_coordinates[:, 0]                                                    
dof_y = dof_coordinates[:, 1]                                                    


## Scatter plot to map discontinuity.
fig = plt.figure()                                                               
ax = fig.add_subplot(111, projection='3d')                                       
ax.scatter(dof_x, dof_y, T.vector()[:], c='b', marker='.')                  
ax.scatter(dof_x, dof_y, Marker.vector()[:], c='r', marker='.')                  
ax.scatter(dof_x, dof_y, Marker1.vector()[:], c='m', marker='.')                  
plt.xlabel("x")
plt.ylabel("y")
plt.show()                                                                       


