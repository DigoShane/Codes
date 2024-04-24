#This is taken from "https://fenicsproject.org/qa/2715/coordinates-u_nodal_values-using-numerical-source-function/"
#and modified for FEniCS 2019.1.0.
#The OP was asking about how he could display nodal values using matplotlib.


from dolfin import *                                                             
import numpy as np                                                               
from mpl_toolkits.mplot3d import Axes3D                                          
import matplotlib.pyplot as plt                                                  

mesh = UnitSquareMesh(10,10)                                         
V = FunctionSpace(mesh, "CG", 2)                                                 

n = V.dim()                                                                      
d = mesh.geometry().dim()                                                        

dof_coordinates = V.tabulate_dof_coordinates().reshape(V.dim(),mesh.geometry().dim())                      
dof_coordinates.resize((n, d))                                                   
dof_x = dof_coordinates[:, 0]                                                    
dof_y = dof_coordinates[:, 1]                                                    

# use FEniCS to get some data to plot                                            
out = Expression("sin(pi*x[0])*cos(pi*x[1])", degree=2)                                    
u = interpolate(out, V)                                                          

fig = plt.figure()                                                               
ax = fig.add_subplot(111, projection='3d')                                       
ax.scatter(dof_x, dof_y, u.vector()[:], c='b', marker='.')                  
plt.show()                                                                       

# now compute the data to be used by FEniCS                                      
z = np.exp(-(dof_x**2 + dof_y**2))                                               
u.vector()[:] = z                                                                

plot(u)      
plt.show()






