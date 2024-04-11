#The objective of this file is to construct a test function for the vortex solution.
#We want to project them into different spaces to make sure that we got the righ figure.

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
r = float(input("r? --> "))

#Create mesh and define function space
mesh = RectangleMesh(Point(0, 0), Point(lx, ly), np.ceil(lx*10/kappa), np.ceil(ly*10/kappa), "crossed") # "crossed means that first it will partition the ddomain into Nx x Ny rectangles. Then each rectangle is divided into 4 triangles forming a cross"
x = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, "Lagrange", 2)#This is for ExtFile

#Plotting \theta = arctan(x)
t = interpolate( Expression('(x[0]-0.5*lx)*(x[0]-0.5*lx) + (x[1]-0.5*ly)*(x[1]-0.5*ly) <= r*r + DOLFIN_EPS ? 1 : atan((x[1]-0.5*ly)/(x[0]-0.5*lx))', lx=lx, ly=ly, r=r, degree=2), V)

#Plotting u(x) = tanh(|x|)
u = interpolate( Expression('tanh(sqrt((x[0]-0.5*lx)*(x[0]-0.5*lx)+(x[1]-0.5*ly)*(x[1]-0.5*ly)))', lx=lx, ly=ly, degree=2), V)

#Plotting A(x) = Piecewise(|x|, Exp[-|x|])
A = interpolate( Expression('sqrt( (x[0]-0.5*lx)*(x[0]-0.5*lx)+(x[1]-0.5*ly)*(x[1]-0.5*ly)) <= r + DOLFIN_EPS ? sqrt((x[0]-0.5*lx)*(x[0]-0.5*lx)+(x[1]-0.5*ly)*(x[1]-0.5*ly)) : \
                             exp(-sqrt((x[0]-0.5*lx)*(x[0]-0.5*lx)+(x[1]-0.5*ly)*(x[1]-0.5*ly)))/K', lx=lx, ly=ly, r=0.3517, K=kappa, degree=2), V)

c = plot(t)
plt.title(r"$\theta(x)$",fontsize=26)
plt.colorbar(c)
plt.show()
c = plot(u)
plt.title(r"$u(x)$",fontsize=26)
plt.colorbar(c)
plt.show()
c = plot(A)
plt.title(r"$A(x)$",fontsize=26)
plt.colorbar(c)
plt.show()





