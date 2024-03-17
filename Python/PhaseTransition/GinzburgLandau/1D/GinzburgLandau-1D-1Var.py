#Here we solve the 1D Ginzbug Landau problem with an applied magnetic field.
# the whole formulation is presented in OneNote UH/superConductivity/Coding/1D Ginzburg Landau fenics.
#This is a 1D version of the problem where A is fixed to Hx_1. This is why the solution comes out to u=0 uniformly. 


import fenics as fe
from dolfin import *
import matplotlib.pyplot as plt

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}

# Create mesh and define function space
Lx=10
mesh = fe.IntervalMesh(16,-Lx,Lx)
#VA = FunctionSpace(mesh, "Lagrange", 2)#this should be a vector.
Vu = FunctionSpace(mesh, "Lagrange", 2)
#Vue = VectorElement('CG', mesh.ufl_cell(), 1) # displacement finite element
#VAe = FiniteElement('CG', mesh.ufl_cell(), 2) # temperature finite element
#V = FunctionSpace(mesh, MixedElement([Vue, Vte]))


# Define functions
u = Function(Vu)
du = TrialFunction(Vu)
v = TestFunction(Vu)
#A = TrialFunction(VA)
#dA = TestFunction(VA)


# Parameters
H = Constant(1);
kappa = Constant(2);
A = Expression("H*x[0]", degree=2)#Solution for normal phase


# Stored strain energy density (compressible neo-Hookean model)
#phi = (1-u**2)**2/2 + (1/kappa**2)*inner(grad(u), grad(u)) + A**2*u**2 + inner(grad(A)-H, grad(A)-H)#Full coupled equation
phi = (1-u**2)**2/2 + (1/kappa**2)*inner(grad(u), grad(u)) + A**2*u**2 #+ inner(H, H)#keeping A fixed. We plug in |\curl A-H|^2 by hand
#!!xDx!!#The weak form is given by
#!!xDx!!#a = (-2*(1-u^2)*u*du + (2/kappa^2)*inner(grad(u), grad(du)) + 2*A^2*u*du + 2*u^2*A*dA + 2*inner(grad(A)-H, grad(dA)))*dx

# Total potential energy
Pi = phi*dx
 
# Compute first variation of Pi (directional derivative about u in the direction of v)
F = derivative(Pi, u, v)

# Compute Jacobian of F
J = derivative(F, u, du)

# Solve variational problem
solve(F == 0, u, J=J,
      form_compiler_parameters=ffc_options)

# Save solution in VTK format
file = File("order.pvd");
file << u;

#pie = assemble((1/(2*Lx))*((1-u**2)**2/2 + (1/kappa**2)*inner(grad(u), grad(u)) + A**2*u**2 + inner(H, H))*dx) #keeping A fixed. We plug in |\curl A-H|^2 by hand
pie = assemble((1/(2*Lx))*((1-u**2)**2/2 + (1/kappa**2)*inner(grad(u), grad(u)) + A**2*u**2)*dx) #keeping A fixed. We plug in |\curl A-H|^2 by hand
print( pie)


#!!xDx!!# Plot and hold solution
#!!xDx!!plot(u, mode = "order", interactive = True)


#!!xDx!!# Visualize
#!!xDx!!c = fe.plot(u, mode="color")
#!!xDx!!plt.colorbar(c)
#!!xDx!!fe.plot(mesh)
#!!xDx!!
#!!xDx!!plt.show()








