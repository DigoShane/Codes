#Here we solve the 1D Ginzbug Landau problem with an applied magnetic field.
#We want to study the transition layer, so we impose boundary conditions of Superconducting on 1 side and Normal on the other.
#We use gradient descent and call it time dependent Ginzburg Landau problem.
# the whole formulation is presented in OneNote UH/superConductivity/Coding/Time dependent Ginzburg Landau Section II
#=====================================================================================================
#THE CODE IS INCOMPLETE.....
#& WE DECIDED TO IGNORE THIS APPROACH.

from dolfin import *
import fenics as fe
import matplotlib.pyplot as plt
import sympy as sym

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"].update( {"representation":"uflacs","log_level": 25, "split": True, "quadrature_degree": 4})
ffc_options = {"optimize": True}

# Create mesh and define function space
Lx=10
T = 2.0            # final time
num_steps = 20     # number of time steps
tau = T / num_steps # time step size

#Defining Function spaces
mesh = fe.IntervalMesh(32,-Lx,Lx)
x = SpatialCoordinate(mesh)
VA = FiniteElement("CG", mesh.ufl_cell(), 2)
Vu = FiniteElement("CG", mesh.ufl_cell(), 2)
V = FunctionSpace(mesh, MixedElement(VA, Vu))

# Parameters
kappa = Constant(2);
H = Constant(0.1);
Ae = H*x[0];


def left(x, on_boundary):
    return near(x[0],0) #and on_boundary
def right(x, on_boundary):
    return near(x[0],Lx) #and on_boundary

bc1 = DirichletBC(V.sub(1), Constant(0.), left)#u(0)=0
bc2 = DirichletBC(V.sub(1), Constant(1.), right)#u(Lx)=0
bc3 = DirichletBC(V.sub(0), Constant(0.), right)#A(Lx)=0
bcs = [bc1, bc2, bc3]

#-----------------------------------------------------------------------------------------------------------------
#!!xDx!! Weak form implicit Euler time stepping
#-----------------------------------------------------------------------------------------------------------------
# Define initial value
Au_n = interpolate( Expression(("H*x[0]","1"), H=0, degree=2), V)#SC phase as initial cond.
(A_n, u_n) = split(Au_n)

# Define functions
Bv = TestFunction(V)
(B, v) = split(Bv)
Au = Function(V)
(A, u) = split(Au)

Fu = ( v*u - tau*v*u*(1-u**2) + (tau/kappa**2)*inner(grad(v), grad(u)) + tau*v*u*A**2 )*dx -u_n*v*dx
FA = ( B*A + tau*inner(grad(B), grad(A)) + tau*B*A*u**2 )*dx -tau*B*H*ds - A_n*B*dx 
F = Fu + FA

# Time-stepping
Au = Function(V)
(A, u) = split(Au)
t = 0
for n in range(num_steps):
    t += tau

    # Compute solution
    #solve(a == L, Au, bcs)
    solve(F == 0, Au, bcs, form_compiler_parameters=ffc_options)
    #solve(F == 0, Au, bcs, solver_parameters={"newton_solver": {"relative_tolerance": 1e-6}})# Solve Equilibrium eqns

    A = Au.sub(0, deepcopy=True)
    u = Au.sub(1, deepcopy=True)

    pie = assemble((1/Lx)*((1-u**2)**2/2 + (1/kappa**2)*inner(grad(u), grad(u)) + A**2*u**2 + inner(grad(A-Ae), grad(A-Ae)))*dx )
    print('Energy density at t=', t, 'is =', pie)
    
    # Plot solution
    plot(u)
    plt.title(r"$u(x)$",fontsize=26)
    plt.show()
    plot(A)
    plt.title(r"$A(x)e_2$",fontsize=26)
    plt.show()

    # Update previous solution
    u_n.assign(u)
    A_n.assign(A)




