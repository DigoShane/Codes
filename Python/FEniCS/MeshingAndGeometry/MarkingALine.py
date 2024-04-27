#the objective of this code is to see if i can mark the midline of a square domain
#as a seperate subdomain.

#import dolfin
#print(f"DOLFIN version: {dolfin.__version__}")
#from dolfin import *
#import fenics as fe
#import numpy as np
#import ufl
#print(f" UFL version: {ufl.__version__}")
#from ufl import tanh
#import matplotlib.pyplot as plt
#
#import sys
#np.set_printoptions(threshold=sys.maxsize)
#
#set_log_level(1)
#
##Parameters
#lx = float(1.0)
#ly = float(1.0)
#
##Create mesh and define function space
#mesh = RectangleMesh(Point(0., 0.), Point(lx, ly), 5, 5, "crossed") # "crossed means that first it will partition the ddomain into Nx x Ny rectangles. Then each rectangle is divided into 4 triangles forming a cross"
#x = SpatialCoordinate(mesh)
#V = FunctionSpace(mesh, "Lagrange", 2)#This is for ExtFile
#
#class Midline(SubDomain):
#    def inside(self, x):
#        return x[1] == ly*0.5 + DOLFIN_EPS
#
#sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
#
#midline = Midline()
##midline.mark(sub_domains,1)
#sub_domains.set_all(1)
#
#file = File("subdomains.pvd")
#file << sub_domains


from dolfin import *
import numpy as np

mesh = UnitSquareMesh(10,10)

class left_side(SubDomain):
    def inside(self, x, on_boundary):
        return np.isclose(x[0], 0.5) 

markers = MeshFunction("size_t", mesh,  mesh.topology().dim()-1, 0)

left_side().mark(markers,1)


ds = ds(domain=mesh, subdomain_data=markers)
print(assemble(1*ds), assemble(1*ds(1)))

File("markers.pvd") << markers


