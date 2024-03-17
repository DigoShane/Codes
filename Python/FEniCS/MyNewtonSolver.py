#this is my own code for a Newton Solver. I am copying the code from 
#"https://jsdokken.com/dolfinx-tutorial/chapter4/newton-solver.html#newtons-method-with-dirichletbc"
#but modified from FEniCSx to FEniCS.
#The problem we solve here is:-
# Ω=[0,1]^2, \Gamma_D=\Gamma_1U\Gamma_2, \Gamma_N= \Gamma_3U\Gamma_4
#\Gamma_1={x=0}, \Gamma_2={y=0}
#\Gamma_3={x=1}, \Gamma_4={y=1}
#div(q(u) ∂u)= f in Ω
#q(u)=1+u^2
#f = -10(1+x+2y)
#u=u_D on \Gamma_D
#u_D\vert_{Gamma_1}=1+2*y
#u_D\vert_{Gamma_2}=1+x
#q(u)∂u/∂n=g on \Gamma_N
#g\vert{\Gamma_3}=q(u*)= 1+(2+2y)^2
#g\vert{\Gamma_4}=2*q(u*)= 2+2(3+x)^2
#We get the solution by using the manufactured soln u*=1+x+2*y 


from dolfin import *
import petsc4py.PETSc as pet
import numpy as np
import matplotlib.pyplot as plt


mesh = UnitSquareMesh(32, 32)
V = FunctionSpace(mesh, "CG", 2)
u = Function(V)
v = TestFunction(V)
deltau = Function(V)
x = SpatialCoordinate(mesh)
max_iterations = 25

#Defining the boundaries
def GammaD1(x, on_boundary):
    tol = 1E-14
    return on_boundary and near(x[0], 0, tol)


def GammaD2(x, on_boundary):
    tol = 1E-14
    return on_boundary and near(x[1], 0, tol)

class GammaD3(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14
        return on_boundary and near(x[0], 1, tol)

class GammaD4(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14
        return on_boundary and near(x[1], 1, tol)

#Applying Dirichletbc's
u_D1 = Expression("1+2*x[1]", degree=1) 
u_D2 = Expression("1+x[0]", degree=1) 
bcs = [DirichletBC(V, u_D1, GammaD1), DirichletBC(V, u_D2, GammaD2)]

#Defining measures for the Neumann BC.
n = FacetNormal(mesh)
gammaD3 = GammaD3()
gammaD4 = GammaD4()
# Initialize mesh function for boundary domains
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundaries.set_all(0)
gammaD3.mark(boundaries, 3)#Marking D3 as 3.
gammaD4.mark(boundaries, 4)#Martking D4 as 4.

ds = Measure("ds", domain=mesh, subdomain_data=boundaries)


f = Expression("-10*(1+x[0]+2*x[1])", degree=1) 
g3 = Expression("1+(2+2*x[1])*(2+2*x[1])", degree=1) 
g4 = Expression("2+2*(3+x[0])*(3+x[0])", degree=1) 


##---------------------------------------------------------------------------
##Default Newton Solver
##---------------------------------------------------------------------------
#F = (1 + u**2)*inner(grad(u), grad(v))*dx - f*v*dx - g3*v*ds(3) -g4*v*ds(4)
#
#solve(F == 0, u, bcs,
#   solver_parameters={"newton_solver":{"relative_tolerance":0.001},"newton_solver":{"maximum_iterations":500}})

#---------------------------------------------------------------------------
#THIS METHOD IS INCOMPLETE. NEED to Finish this.
#Self written Newton Solver.
#---------------------------------------------------------------------------
F = (1 + u**2)*inner(grad(u), grad(v))*dx - f*v*dx - g3*v*ds(3) -g4*v*ds(4)
J = derivative(F,u)
#residual = dolfin.fem.form(F)
residual = fem.form.Form(F)
#F'(x_n)dx_n=-F(x_n), where dx_n=x_n+1-x_n.
A = PETScMatrix()
b = PETScVector()
L = PETScVector()
A = assemble(J, tensor = A);
b = assemble(-F, tensor = b);
L = assemble(residual, tensor = b);
#solver = PETScKrylovSolver(ksp)
solver = PETScKrylovSolver("default","default")
solver.set_operator(A)

i=0
#coords = V.tabulate_dof_coordinates(mesh)[:, 0]
coords = V.tabulate_dof_coordinates()[:, 0]
sort_order = np.argsort(coords)
solutions = np.zeros((max_iterations + 1, len(coords)))
#solutions[0] = u.x.array[sort_order]
solutions[0] = u.array[sort_order]

#i = 0
#while i < max_iterations:
#    # Assemble Jacobian and residual
#    with L._Vec_localForm() as loc_L:
#        loc_L.set(0)
#    A.zeroEntries()
#    dolfinx.fem.petsc.assemble_matrix(A, jacobian)
#    A.assemble()
#    dolfinx.fem.petsc.assemble_vector(L, residual)
#    L.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
#
#    # Scale residual by -1
#    L.scale(-1)
#    L.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)
#
#    # Solve linear problem
#    solver.solve(L, dx.vector)
#    dx.x.scatter_forward()
#    # Update u_{i+1} = u_i + delta x_i
#    uh.x.array[:] += dx.x.array
#    i += 1
#
#    # Compute norm of update
#    correction_norm = dx.vector.norm(0)
#    print(f"Iteration {i}: Correction norm {correction_norm}")
#    if correction_norm < 1e-10:
#        break
#    solutions[i, :] = uh.x.array[sort_order]
















plt.figure()
plot(u, title="Solution")

plt.figure()
plot(grad(u), title="Solution gradient")

plt.show()

