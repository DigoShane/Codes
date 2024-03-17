#Here we solve the 1D Ginzbug Landau problem with an applied magnetic field.
#the whole formulation is presented in OneNote UH/superConductivity/Coding/1D Ginzburg Landau fenics.
# take a look at SEC-IV.
#This is a 1D version of the problem where A is not fixed

from dolfin import *
import fenics as fe
import matplotlib.pyplot as plt
from ufl import tanh

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}

# Create mesh and define function space
Lx=5# here the domain is (0,Lx)
mesh = fe.IntervalMesh(128,0,Lx)
x = SpatialCoordinate(mesh)
VA = FiniteElement("CG", mesh.ufl_cell(), 2)
Vu = FiniteElement("CG", mesh.ufl_cell(), 2)
V = FunctionSpace(mesh, MixedElement(VA, Vu))

# Define functions
Bv = TestFunction(V)
(B, v) = split(Bv)
dAu = TrialFunction(V)
(dA, du) = split(dAu)


# Parameters
kappa = Constant(10);
 

Au = interpolate( Expression(("H*x[0]","1"), H=0, degree=20), V)

#Dirichlet BC for left bdry
def boundary_L(x, on_boundary):
    tol = 1E-24
    return on_boundary and near(x[0], 0, tol)
def boundary_R(x, on_boundary):
    tol = 1E-24
    return on_boundary and near(x[0], Lx, tol)

bc1 = DirichletBC(V.sub(1), 0, boundary_L)
bc2 = DirichletBC(V.sub(1), 1, boundary_R)
bc3 = DirichletBC(V.sub(0), 0, boundary_R)

bcs = [bc1, bc2, bc3];

for i in range(1,10,1):
 H = Constant(i/10);
 Ae = H*x[0];
 #Compute first variation of Pi (directional derivative about u in the direction of v)
 #Au = interpolate( Expression(("H*x[0]","1"), H=0, degree=2), V)
 (A, u) = split(Au)
 ##----------------------------------------------------------------
 #plot(u)
 #plt.title(r"$B4-For H= %s$" %(i/10),fontsize=26)
 #plt.ylabel('u')
 #plt.xlabel('x')
 #plt.show()
 #plot(A)
 #plt.ylabel('A.e_2')
 #plt.xlabel('x')
 #plt.title(r"$B4-For H= %s$" %(i/10),fontsize=26)
 #plt.show()
 ##----------------------------------------------------------------
 F = (-(1-u**2)*u*du + (1/kappa**2)*inner(grad(u), grad(du)) + A**2*u*du + u**2*A*dA + inner(grad(A), grad(dA)))*dx + H*dA*ds
 solve(F == 0, Au, bcs,
   solver_parameters={"newton_solver":{"convergence_criterion":"incremental","relaxation_parameter":0.01,"relative_tolerance":0.000001,"absolute_tolerance":0.001,"maximum_iterations":500}})


 A = Au.sub(0, deepcopy=True)
 u = Au.sub(1, deepcopy=True)
 
 pie = assemble((1/Lx)*((1-u**2)**2/2 + (1/kappa**2)*inner(grad(u), grad(u)) + A**2*u**2 + inner(grad(A-Ae), grad(A-Ae)))*dx )
 print("The energy for (H=",H,") =",pie)

 #----------------------------------------------------------------- 
 plot(u)
 plt.title(r"$For H= %s$" %(i/10),fontsize=26)
 plt.ylabel('u')
 plt.xlabel('x')
 plt.show()
 plot(A)
 plt.ylabel('A.e_2')
 plt.xlabel('x')
 plt.title(r"$For H= %s$" %(i/10),fontsize=26)
 plt.show()
#----------------------------------------------------------------- 
# #plot(u, label="at H=%s" %(i/10))
# plot(A, label="at H=%s" %(i/10))
#plt.legend(loc="upper left")
#plt.show()
##----------------------------------------------------------------- 
