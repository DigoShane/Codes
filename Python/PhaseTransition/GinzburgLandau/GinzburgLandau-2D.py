#Here we solve the 2D Ginzbug Landau problem with an applied magnetic field along e_3.
# the whole formulation is presented in OneNote UH/superConductivity/Coding/1D Ginzburg Landau fenics.
#This is a 1D version of the problem where A is not fixed
#=====================================================================================================
#THE CODE IS INCOMPLETE.....

from dolfin import *
import fenics as fe
import matplotlib.pyplot as plt

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}

# Create mesh and define function space
Lx=10
mesh = fe.IntervalMesh(32,-Lx,Lx)
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
kappa = Constant(2);
H = Constant(0.748);
Ae = H*x[0];



#-----------------------------------------------------------------------------------------------------------------
#!!xDx!! ##!!xDx!! Newton rhapson Approach
#-----------------------------------------------------------------------------------------------------------------
#Compute first variation of Pi (directional derivative about u in the direction of v)
#Au = Function(V)
Au = interpolate( Expression(("H*x[0]","1"), H=0, degree=2), V)#SC phase as initial cond.
#Au = interpolate( Expression(("H*x[0]","0"), H=H, degree=2), V)#Normal phase as initial condiiton
(A, u) = split(Au)
F = (-2*(1-u**2)*u*du + (2/kappa**2)*inner(grad(u), grad(du)) + 2*A**2*u*du + 2*u**2*A*dA + 2*inner(grad(A-Ae), grad(dA)))*dx
solver.parameters.nonzero_initial_guess = True
solve(F == 0, Au, bcs,
   solver_parameters={"newton_solver":{"convergence_criterion":"incremental","relaxation_parameter":0.01,"relative_tolerance":0.000001,"absolute_tolerance":0.001,"maximum_iterations":500}})

A = Au.sub(0, deepcopy=True)
u = Au.sub(1, deepcopy=True)

## Save solution in VTK format
#file = File("order-A.pvd");
#file << A;
#file = File("order-u.pvd");
#file << u;

pie = assemble((1/(2*Lx))*((1-u**2)**2/2 + (1/kappa**2)*inner(grad(u), grad(u)) + A**2*u**2 + inner(grad(A-Ae), grad(A-Ae)))*dx )
print(pie)

plot(u)
plt.title(r"$u(x)$",fontsize=26)
plt.show()
plot(A)
plt.title(r"$A(x)e_2$",fontsize=26)
plt.show()




#This was commented cause it didnt really work well.
##!!cDc!!#-----------------------------------------------------------------------------------------------------------------
##!!cDc!!##!!xDx!!EnergyMinimization Approach
##!!cDc!!#-----------------------------------------------------------------------------------------------------------------
##!!cDc!!# Helmholtz free energy density.
##!!cDc!!phi = (1-u**2)**2/2 + (1/kappa**2)*inner(grad(u), grad(u)) + A**2*u**2 + inner(grad(A-Ae), grad(A-Ae))
##!!cDc!!Pi = phi*dx
##!!cDc!!F = derivative(Pi, Au, Bv)# Compute first variation
##!!cDc!!J = derivative(F, Au, dAu)# Compute Jacobian of F
##!!cDc!!M = inner(Au,Au)*dx
##!!cDc!!Au = interpolate( Expression(("1","1"), degree=1), V)
##!!cDc!!tol = 1.e-9
##!!cDc!!# Solve variational problem
##!!cDc!!problem=NonlinearVariationalProblem(F, Au,J=J)
##!!cDc!!solver = AdaptiveNonlinearVariationalSolver(problem, M)
##!!cDc!!solver.parameters['nonlinear_variational_solver']["nonlinear_solver"] = 'snes'
##!!cDc!!#solver.parameters.nonzero_initial_guess = True
##!!cDc!!solver.solve(tol)


