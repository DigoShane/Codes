#Implement inequality constriants in FEniCS.

from fenics import *
#from fenics_adjoint import *
import matplotlib.pyplot as plt

try:
    from pyadjoint import ipopt  # noqa: F401
except ImportError:
    print("""This example depends on IPOPT and Python ipopt bindings. \
  When compiling IPOPT, make sure to link against HSL, as it \
  is a necessity for practical problems.""")
    raise

# turn off redundant output in parallel
parameters["std_out_all_processes"] = False

flag_showError1 = False
flag_showError2 = False


Vlim = Constant(0.4)  # volume bound on the control
Qlim = 1e-4
p = Constant(5)  # power used in the solid isotropic material with penalisation (SIMP) rule, to encourage the control solution to attain either 0 or 1
eps = Constant(1.0e-3)  # epsilon used in the solid isotropic material
alpha = Constant(1.0e-8)  # regularisation coefficient in functional


def k(a):
    return eps + (1 - eps) * a ** p


n = 250
mesh = UnitSquareMesh(n, n)
A = FunctionSpace(mesh, "CG", 1)  # function space for control
P = FunctionSpace(mesh, "CG", 1)  # function space for solution

class WestNorth(SubDomain):
    """The top and left boundary of the unitsquare, used to enforce the Dirichlet boundary condition."""
    def inside(self, x, on_boundary):
        return (x[0] == 0.0 or x[1] == 1.0) and on_boundary


bc = [DirichletBC(P, 0.0, WestNorth())] # the Dirichlet BC; the Neumann BC will be implemented implicitly by dropping the surface integral after integration by parts
f = interpolate(Constant(1.0e-2), P)  # the volume source term for the PDE

def forward(a):
    """Solve the forward problem for a given material distribution a(x)."""
    T = Function(P)
    v = TestFunction(P)
    F = inner(grad(v), k(a) * grad(T)) * dx - f * v * dx
    solve(F == 0, T, bc, solver_parameters={"newton_solver": {"absolute_tolerance": 1.0e-7, "maximum_iterations": 20}})
    return T

    
#-------------------------------------------------
class CustomConstraint(InequalityConstraint):
    """A class that enforces the volume constraint g(a) = V - a*dx >= 0."""
    def __init__(self, Qlim):
        self.Qlim = float(Qlim)
        self.tmpvec = Function(A)

    def function(self, m):
        from pyadjoint.reduced_functional_numpy import set_local
        set_local(self.tmpvec, m)
        
        T = forward(self.tmpvec)
        Q = assemble(inner(grad(T),k(self.tmpvec)*grad(T)) * dx)
        return [-self.Qlim + Q]

    def jacobian(self, m):
        from pyadjoint.reduced_functional_numpy import set_local
        set_local(self.tmpvec, m)
        T = forward(self.tmpvec)
        Q = assemble(inner(grad(T),k(self.tmpvec)*grad(T)) * dx)
        dQdm = compute_gradient(Q, m)
        return [dQdm]

    def output_workspace(self):
        return [0.0]

    def length(self):
        """Return the number of components in the constraint vector (here, one)."""
        return 1
#-------------------------------------------------

if __name__ == "__main__":
    a = interpolate(Vlim, A)  # initial guess.
    T = forward(a)  # solve the forward problem once.
    
    J = assemble(f * T * dx + alpha * inner(grad(a), grad(a)) * dx)
    m = Control(a)
    Jhat = ReducedFunctional(J, m)
    
    lb = 0.0; ub = 1.0
    
    volume_constraint = UFLInequalityConstraint((Vlim - a)*dx, m)
    
    if flag_showError1:
        losses_constraint = UFLInequalityConstraint((-Qlim + inner(grad(forward(T)),k(T)*grad(forward(T))))*dx, m)
        problem = MinimizationProblem(Jhat, bounds=(lb, ub), constraints=[volume_constraint,losses_constraint])
    elif flag_showError2:
        problem = MinimizationProblem(Jhat, bounds=(lb, ub), constraints=[volume_constraint,CustomConstraint(Qlim)])
    else:
        problem = MinimizationProblem(Jhat, bounds=(lb, ub), constraints=volume_constraint)


    solver = IPOPTSolver(problem, parameters={"acceptable_tol": 1.0e-3, "maximum_iterations": 100})
    a_opt = solver.solve()
    T_opt = forward(a_opt)
    
    pl, ax = plt.subplots(); fig = plt.gcf(); fig.set_size_inches(16, 4)
    plt.subplot(1, 2, 1); p = plot(a_opt,title='Topology',mode='color',vmin=0,vmax=1.0); p.set_cmap("Greys"); cbar = plt.colorbar(p); 
    plt.subplot(1, 2, 2); p = plot(T_opt,title='Temperature',mode='color'); p.set_cmap("coolwarm"); cbar = plt.colorbar(p);
