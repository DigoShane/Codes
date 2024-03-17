#Here we solve the 1D Ginzbug Landau problem with an integral constraint with an applied magnetic field.
#the basic formulation is presented in OneNote UH/superConductivity/Coding/1D Ginzburg Landau fenics.
# take a look at SEC-IV.
#We loop over the applied magnetic fields.
#========================================================================================================
#Issues with different approaches:-
#1. Using a fixed unevolving constraint throught the simulation.
#   The main issue with this code is that it loops over different fields to get the solution while 
#   imposing the constraint ∫(u-1/2)=0, we are getting the wrong solution for the case of H small.
#   This is because for H small we should be getting the bulk superconducting phase barring edge effects.
#   This does not satisfy the above constraint. If we iterate over H using the output of the previous 
#   field as input, then we end up with the wrong solution at the critical field.
#--------------------------------------------------------------------------------------------------------
#We got rid of the above issue by modifying the code such that the constraint varies with each field.
#========================================================================================================

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
Lx=250# here the domain is (0,Lx)
mesh = fe.IntervalMesh(1000,0,Lx)
x = SpatialCoordinate(mesh)
VA = FiniteElement("CG", mesh.ufl_cell(), 2)
Vu = FiniteElement("CG", mesh.ufl_cell(), 2)
R = FiniteElement("Real", mesh.ufl_cell(), 0)
V = FunctionSpace(mesh, MixedElement(VA, Vu, R))

# Define functions
dAur = TrialFunction(V)
(dA, du, dr) = split(dAur)

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

##Specifying the intiial conditions.
#Coexistence of phase as initial condition
ul = Expression('0', degree=2, domain=mesh)
Al = Expression('0', degree=2, domain=mesh)
#Al = Expression('H*(0.5*Lx-x[0])', H=0, Lx=Lx, degree=2, domain=mesh)
ur = Expression('1', degree=2, domain=mesh)
Ar = Expression('0', degree=2, domain=mesh)
Aur = interpolate( Expression(('x[0] <= 0.5*Lx + DOLFIN_EPS ? Al : Ar', 'x[0]<=0.5*Lx+DOLFIN_EPS ? ul : ur', '11'), ul=ul, ur=ur, Al=Al, Ar=Ar, Lx=Lx, degree=2), V)

# Parameters
kappa = Constant(2);
f = Constant(0.5); #The constraint is ∫(u-f)dx=0
n = 100 #This is the number of times after which i will stop the simulation and print u(x) and A(x)

for i in range(1,1000,1):
 H = Constant(i/1000);
 Ae = H*x[0];
 (A, u, r) = split(Aur)
 
 print("===============================")
 print("the calculation for (H=%0.4f) ="%H)
 print("===============================")
 F = (-(1-u**2)*u*du + (1/kappa**2)*inner(grad(u), grad(du)) + A**2*u*du + 0.5*r*du + dr*(u-f) + u**2*A*dA + inner(grad(A), grad(dA)))*dx + H*dA*ds
 #solver.parameters.nonzero_initial_guess = True
 solve(F == 0, Aur, bcs,
   solver_parameters={"newton_solver":{"convergence_criterion":"incremental","relaxation_parameter":0.01,"relative_tolerance":0.000001,"absolute_tolerance":0.001,"maximum_iterations":500}})
 
 A = Aur.sub(0, deepcopy=True)
 u = Aur.sub(1, deepcopy=True)
 r = Aur.sub(2, deepcopy=True)
 
 pie = assemble((1/(Lx))*((1-u**2)**2/2 + (1/kappa**2)*inner(grad(u), grad(u)) + A**2*u**2 + inner(grad(A-Ae), grad(A-Ae)))*dx )
 print("The energy for (H=%0.4f) ="%H,pie)
 Constraint = assemble( (u-f)*dx)
 print("Constraint violated for (H=%0.4f) by ="%H, Constraint)

 if i%n==0:
  plot(u)
  plt.title(r"$For H= %.4f$" %H,fontsize=26)
  plt.ylabel('u')
  plt.xlabel('x')
  plt.show()
  plot(A)
  plt.ylabel('A.e_2')
  plt.xlabel('x')
  plt.title(r"$For H= %.4f$" %H,fontsize=26)
  plt.show()
