#The objective of this code is to return the normal to the boundary
#I used the code given in "https://fenicsproject.org/qa/13004/dirichlet-condition-with-normal-vector/"

#from dolfin import *
#import numpy as np
#import matplotlib.pyplot as plt
#
#mesh = UnitSquareMesh(10, 10)
#bmesh = BoundaryMesh(mesh, 'exterior', True)
#n = FacetNormal(mesh)
#
#plt.figure()
#plot(n, title="normal")
#plt.show()
#
#fid = File('normal.pvd')
#fid << n
#
#
#
#

##=====================================================================================================================
#from dolfin import *
#
#
##create mesh and define function space
#length=width=5.0
#height = 1.0
#mesh = BoxMesh(Point(0., 0., 0.), Point(length, width, height), 10, 10, 2)
#V = VectorFunctionSpace(mesh, 'P', 1)
#
#loop_side = 1.0 #length of square loop
#C = 0.1 #constant (~Remanance/4pi) for boundary eqn component
#tol = 1E-14 #tolerance for boundary definition
#loop_height_z = 0.5 #the sq loop will be at this height
#
#class bndry(SubDomain):
#    def inside(self, x, on_boundary):
#        return on_boundary and x[0] <= length/2 and x[1] < width/2 and x[2]< 0.5*height
#facetfunc = MeshFunction("size_t", mesh, mesh.geometry().dim()-1)
#bnd = bndry()
#bnd.mark(facetfunc, 3)
#
#File("facetfunc.pvd").write(facetfunc)

#=====================================================================================================================
from dolfin import *
import matplotlib.pyplot as plt
import mshr


#Create mesh and define function space
l = 10
r = 5
domain = mshr.Rectangle(Point(-l,-l), Point(l, l)) - mshr.Circle(Point(0., 0.), r)
mesh = mshr.generate_mesh(domain, int(l)*10)
x = SpatialCoordinate(mesh)
Va1 = FiniteElement("CG", mesh.ufl_cell(), 2)
Va2 = FiniteElement("CG", mesh.ufl_cell(), 2)
Vu = FiniteElement("CG", mesh.ufl_cell(), 2)
V = FunctionSpace(mesh, MixedElement(Va1, Va2, Vu))
Vcoord = FunctionSpace(mesh, "Lagrange", 2)#This is for ExtFile


#Boundary conditions
class Inner(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0]**2+x[1]**2, r, DOLFIN_EPS)

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1] , l , DOLFIN_EPS) 

class Left(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0] , -l , DOLFIN_EPS) 

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1] , -l , DOLFIN_EPS) 

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0] , l , DOLFIN_EPS) 



dom = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
iinner = Inner()
top = Top()
left = Left()
bottom = Bottom()
right = Right()

top.mark(boundaries, 2)
left.mark(boundaries, 3)
bottom.mark(boundaries, 4)
right.mark(boundaries, 5)
iinner.mark(boundaries, 1) 
dom

ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

File("boundaries.pvd").write(boundaries)


