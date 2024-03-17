#================================================================================================================
#              Reason for this code
#================================================================================================================
#The reason for this code is to understand how to define piecewise functions when we have a tuple of functions
#The problem set up here is the vector Poisson equation.
#The formulation being used is the plate with a hole.
#================================================================================================================
#ISSUES WITH THE CODE
#I cannot seem to get the piece-wise defined vector function to work.
#the ufl_shape of f is non existent which is weird cause we are defining p1 and p2 as vectors properly.
#================================================================================================================
#ALTERNATIVE ROUTES 2 PROCEED with
#check out the subdomain function, that might be helpfuli
#Titled "Defining subdomains for different materials" at the url
#"https://jsdokken.com/dolfinx-tutorial/chapter3/subdomains.html"
#There is also the tutorial labelled
#"Subdomains and boundary conditions"
#where they have a piecewise defined boundary flux.
#located at "https://fenicsproject.org/pub/tutorial/sphinx1/._ftut1005.html".

from dolfin import *
#from ufl import *
#from ufl.classes import *
#from ufl.algorithms import *
import fenics as fe
import matplotlib.pyplot as plt
import numpy as np

L = 2.
H = 2.
Nx = 250
Ny = 10
mesh = RectangleMesh(Point(0., 0.), Point(L, H), Nx, Ny, "crossed")
x = SpatialCoordinate(mesh)

def eps(v):
    return sym(grad(v))

E = Constant(1e5)
nu = Constant(0.3)
model = "plane_stress"
mu = E/2/(1+nu)
lmbda = E*nu/(1+nu)/(1-2*nu)
if model == "plane_stress":
    lmbda = 2*mu*lmbda/(lmbda+2*mu)

def sigma(v):
    return lmbda*tr(eps(v))*Identity(2) + 2.0*mu*eps(v)

#Defining the function space and the functions.
V = VectorFunctionSpace(mesh, 'Lagrange', degree=2, dim=2)
du = TrialFunction(V)
u_ = TestFunction(V)
f = TestFunction(V)
rho_g1 = Constant(1e-3)
rho_g2 = Constant(1)
##piecewise fn defn in 1 shot
#p1 = Expression(("0","c1"), c1=rho_g1, degree=2, domain=mesh)
#p2 = Expression(("c2","0"), c2=rho_g2, degree=2, domain=mesh)
#f = Expression('x[0] <= 0.2 + DOLFIN_EPS ? p1 : p2', p1=p1, p2=p2, degree=2, domain=mesh)
#piecewise fn defn, component wise.
p11 = Expression('0', degree=2, domain=mesh)
p12 = Expression('c1', c1=rho_g1, degree=2, domain=mesh)
p21 = Expression('c2', c2=rho_g2, degree=2, domain=mesh)
p22 = Expression('0', degree=2, domain=mesh)
f = Expression(('x[0] <= 0.2 + DOLFIN_EPS ? p11 : p12', 'x[0]<=0.2+DOLFIN_EPS ? p21 : p22'), p11=p11, p12=p12, p21=p21, p22=p22, degree=2, domain=mesh)
#!!xDx!!#general function
#!!xDx!!f = Expression(("c1","c2"), c1=rho_g1, c2=rho_g2, degree=2, domain=mesh)
print("shape of f=",f.ufl_shape)
print("shape of u_=",u_.ufl_shape)
a = inner(sigma(du), eps(u_))*dx
l = inner(f, u_)*dx

def left(x, on_boundary):
    return near(x[0], 0.)

bc = DirichletBC(V, Constant((0.,0.)), left)
u = Function(V, name="Displacement")
solve(a == l, u, bc)

#Plot solution
plot(f[0])
plt.title(r"$f_1(x)$",fontsize=26)
plt.show()
plot(f[1])
plt.title(r"$f_2(x)$",fontsize=26)
plt.show()
plot(u[0])
plt.title(r"$u_1(x)$",fontsize=26)
plt.show()
plot(u[1])
plt.title(r"$u_2(x)$",fontsize=26)
plt.show()
