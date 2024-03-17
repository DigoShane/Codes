#Here we solve the 1D Ginzbug Landau problem with an applied magnetic field.
#We use the "NonLinearVariationalProblemSolver" option in FEniCS.
#the whole formulation is presented in OneNote UH/superConductivity/Coding/1D Ginzburg Landau fenics.
#look specifically at sec-VI
#The domain for the problem is [0,Lx] for initial conditions for "bulk SC", "bulk Normal", "Coexistence of Phase"
#=====================================================================================================
#ISSUES WITH THE CODE:-
#The code doesnt seem to want to run, also it looks like they are solving F==0. Which suggests that the original code may be the best way to proceed.

from dolfin import *
from ufl import tanh
import fenics as fe
import matplotlib.pyplot as plt

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}

# Create mesh and define function space
Lx=250
mesh = fe.IntervalMesh(1000,0,Lx)
x = SpatialCoordinate(mesh)
VA = FiniteElement("CG", mesh.ufl_cell(), 2)
Vu = FiniteElement("CG", mesh.ufl_cell(), 2)
V = FunctionSpace(mesh, MixedElement(VA, Vu))

# Define functions
Au = Function(V)
dAu = TrialFunction(V)
Bv = TestFunction(V)
(A, u) = split(Au)
(dA, du) = split(dAu)
(B, v) = split(Bv)


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

# Parameters
kappa = Constant(1);
Hin = input("External Magnetic field? ")
H = Constant(Hin);
Ae = H*x[0]


#-----------------------------------------------------------------------------------------------------------------
#!!xDx!! ##!!xDx!! Newton rhapson Approach
#-----------------------------------------------------------------------------------------------------------------
A = Au.sub(0, deepcopy=True)
u = Au.sub(1, deepcopy=True)
# Stored strain energy density (compressible neo-Hookean model)
psi = 0.5*(1-u**2)**2 + (1/kappa**2)*inner(grad(u),grad(u)) + A**2*u**2 + inner(grad(A-Ae),grad(A-Ae))

# Compute first variation of Pi (directional derivative about u in the direction of v)
F = derivative(psi, Au, Bv)

# Compute Jacobian of F
J = derivative(F, Au, dAu)

#initial conditions
#Au = interpolate( Expression(("0","1.5"), degree=2), V)#SC phase as initial cond.
#Au = interpolate( Expression(("H*x[0]","0"), H=H, degree=2), V)#Normal phase as initial condiiton
#Au = interpolate( Expression(("0.5*h*x[0]","0.5*tanh(x[0]-0.5*l)+0.5"), l=Lx, h=H,  degree=10), V)#coexistence of phase as initial cond.
#Coexistence of phase as initial condition
ul = Expression('0', degree=2, domain=mesh)
Al = Expression('H*x[0]', H=H, degree=2, domain=mesh)
ur = Expression('1', degree=2, domain=mesh)
Ar = Expression('0', degree=2, domain=mesh)
Au = interpolate( Expression(('x[0] <= 0.5*Lx + DOLFIN_EPS ? Al : Ar', 'x[0]<=0.5*Lx+DOLFIN_EPS ? ul : ur'), ul=ul, ur=ur, Al=Al, Ar=Ar, Lx=Lx, degree=2), V)
(A, u) = split(Au)

# Solve variational proble
problem = NonlinearVariationalProblem(F, Au, bcs, J)
solver  = NonlinearVariationalSolver(problem)
prm = solver.parameters
prm['newton_solver']['absolute_tolerance'] = 1E-8
prm['newton_solver']['relative_tolerance'] = 1E-6
prm['newton_solver']['convergence_criterion'] = "incremental"
prm['newton_solver']['maximum_iterations'] = 500
prm['newton_solver']['relaxation_parameter'] = 0.3
set_log_level(PROGRESS)

#solve(F == 0, Au, bcs, J=J, form_compiler_parameters=ffc_options)

A = Au.sub(0, deepcopy=True)
u = Au.sub(1, deepcopy=True)

pie = assemble((1/(Lx))*((1-u**2)**2/2 + (1/kappa**2)*inner(grad(u), grad(u)) + A**2*u**2 + inner(grad(A-Ae), grad(A-Ae)))*dx )
print(pie)

plot(u)
plt.title(r"$u(x)$",fontsize=26)
plt.show()
plot(A)
plt.title(r"$A(x)e_2$",fontsize=26)
plt.show()


