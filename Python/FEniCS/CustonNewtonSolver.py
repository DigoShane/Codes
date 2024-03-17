#The objective of this code is to test the Custom Newton Solver
#as reproduced from "https://fenicsproject.discourse.group/t/set-krylov-linear-solver-paramters-in-newton-solver/1070/3"
#The original NewtonSolver seems a bit too rigid and it was recommended to write a new one by myself.
#The NonLinear poisson problem being described here can be found in "https://fenicsproject.org/pub/tutorial/html/._ftut1007.html"


import matplotlib.pyplot as plt
from dolfin import *

#Defining a class called problem which has attributes bilinear_form, linear_form and bcs.
class Problem(NonlinearProblem):
    def __init__(self, J, F, bcs):#A Constructor this allows easy initialization of objects of this type.
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
    #this is a way of calling the constructor of the Newton Solver ParentClass.
    #the arguments of the constructor are 
    #MPI_Comm comm --> mesh.mpi_comm()
    #std::shared_ptr<GenericLinearSolver> solver  --> PETScKrylovSolver()
    #GenericLinearAlgebraFactory& factory --> PETScFactory.instance()
    #??xDx?? Not sure what .instance() is, couldnt find it from a precursory glance in google.

    def solver_setup(self, A, P, problem, iteration):
        self.linear_solver().set_operator(A)

        PETScOptions.set("ksp_type", "gmres")
        PETScOptions.set("ksp_monitor")
        PETScOptions.set("pc_type", "ilu")

        self.linear_solver().set_from_options()

    #This is an example of plymorphism used to override the method "converged" of the parent class "NewtonSolver".
    #"NewtonSolver" has a method called "converged"
    def converged(self, r, problem, iteration):
        if iteration == 0:
            self.r0 = r.norm("l2")
        print(f"Iteration {iteration}, relative residual {r.norm('l2')/self.r0:.6e}")
        return super().converged(r, problem, iteration)


mesh = UnitSquareMesh(32, 32)

V = FunctionSpace(mesh, "CG", 2)

##Case I
#g = Constant(1.0) 
#bcs = [DirichletBC(V, g, "near(x[0], 1.0) and on_boundary")]
#Case II
g = Expression('1+x[0]+2*x[1]', degree=2) #Case II, manufactured using u=1+x+2y
bcs = [DirichletBC(V, g, "on_boundary")]

u = Function(V)
v = TestFunction(V)
#f = Expression("x[0]*sin(x[1])", degree=2) #Case I
f = Expression("-10*(1+x[0]+2*x[1])", degree=1) #Case II, manufactured using u=1+x+2y
F = (1 + u**2)*inner(grad(u), grad(v))*dx - f*v*dx
J = derivative(F, u)

problem = Problem(J, F, bcs)
custom_solver = CustomSolver()
custom_solver.solve(problem, u.vector())

print(dir(problem))
print(dir(custom_solver))


plt.figure()
plot(u, title="Solution")

plt.figure()
plot(grad(u), title="Solution gradient")

plt.show()

