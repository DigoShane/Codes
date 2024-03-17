#Here we solve the 1D Ginzbug Landau problem with an applied magnetic field.
#the whole formulation is presented in OneNote UH/superConductivity/Coding/1D Ginzburg Landau fenics.
#look specifically at sec-IV
#The code is exactly the same formulation as "GinzburgLandau-1D.py". We just write a custom Newton
#Solver here based on 
#"https://fenicsproject.discourse.group/t/set-krylov-linear-solver-paramters-in-newton-solver/1070/3"
#and "https://fenicsproject.discourse.group/t/return-relative-residual-from-newton-solver/7737".
#=====================================================================================================
#ISSUES WITH THE CODE:-

from dolfin import *
from ufl import tanh
import fenics as fe
import matplotlib.pyplot as plt


#-----------------------------------------------------------------------------------------------------------------
#!!xDx!! Custom Newton Rhapson Solver
#-----------------------------------------------------------------------------------------------------------------
class Problem(NonlinearProblem):
    def __init__(self, J, F, bcs):
        self.bilinear_form = J
        self.linear_form = F
        self.bcs = bcs
        NonlinearProblem.__init__(self)

    def F(self, b, x):
        assemble(self.linear_form, tensor=b)
        for bc in self.bcs:
            bc.apply(b, x)

    def J(self, A, x):
        assemble(self.bilinear_form, tensor=A)
        for bc in self.bcs:
            bc.apply(A)


class CustomSolver(NewtonSolver):
    def __init__(self):
        NewtonSolver.__init__(self, mesh.mpi_comm(),
                              PETScKrylovSolver(), PETScFactory.instance())

    def solver_setup(self, A, P, problem, iteration):
        self.linear_solver().set_operator(A)

        PETScOptions.set("ksp_type", "gmres")
        PETScOptions.set("ksp_monitor")
        PETScOptions.set("pc_type", "ilu")

        self.linear_solver().set_from_options()

    def converged(self, r, problem, iteration):
        if iteration == 0:
            self.r0 = r.norm("l2")
        print(f"Iteration {iteration}, relative residual {r.norm('l2')/self.r0:.6e}")
        return super().converged(r, problem, iteration)



#-----------------------------------------------------------------------------------------------------------------
#!!xDx!! Domain and function space description.
#-----------------------------------------------------------------------------------------------------------------
# Create mesh and define function space
Lx=250
mesh = fe.IntervalMesh(500,0,Lx)
x = SpatialCoordinate(mesh)
VA = FiniteElement("CG", mesh.ufl_cell(), 10)
Vu = FiniteElement("CG", mesh.ufl_cell(), 10)
V = FunctionSpace(mesh, MixedElement(VA, Vu))

# Define functions
dAu = TrialFunction(V)
(dA, du) = split(dAu)


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
#Au = interpolate( Expression(("0","1.5"), degree=2), V)#SC phase as initial cond.
#Au = interpolate( Expression(("H*x[0]","0"), H=H, degree=2), V)#Normal phase as initial condiiton
#Coexistence of phase as initial condition
ul = Expression('0', degree=10, domain=mesh)
Al = Expression('H*x[0]', H=H, degree=10, domain=mesh)
ur = Expression('1', degree=10, domain=mesh)
Ar = Expression('0', degree=10, domain=mesh)
Au = interpolate( Expression(('x[0] <= 0.5*Lx + DOLFIN_EPS ? Al : Ar', 'x[0]<=0.5*Lx+DOLFIN_EPS ? ul : ur'), ul=ul, ur=ur, Al=Al, Ar=Ar, Lx=Lx, degree=2), V)
(A, u) = split(Au)

F = (-(1-u**2)*u*du + (1/kappa**2)*inner(grad(u), grad(du)) + A**2*u*du + u**2*A*dA + inner(grad(A), grad(dA)))*dx + H*dA*ds
J = derivative(F, Au)


problem = Problem(J, F, bcs)
custom_solver = CustomSolver()
custom_solver.solve(problem, Au.vector())

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


