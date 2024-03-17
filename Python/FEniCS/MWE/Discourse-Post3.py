#This is a MWE for a post to discourse so that i cna figure out a way to handle multiple roots.
#I retain just 1 order parameter for simplicity.

from dolfin import *
import fenics as fe
import numpy as np
import matplotlib.pyplot as plt


#Create mesh and define function space
Lx=10
mesh = fe.IntervalMesh(100,0,Lx)
x = SpatialCoordinate(mesh)
Vw = FiniteElement("CG", mesh.ufl_cell(), 2) #Element for u
R = FiniteElement("Real", mesh.ufl_cell(), 0) #Element for Lagrange multiplier
V = FunctionSpace(mesh, MixedElement(Vw, R)) #Creating functionSpace
VFnSp = FunctionSpace(mesh, "Lagrange", 2)
RFnSp = FunctionSpace(mesh, "Real", 0)


#Dirichlet BC for left bdry
def boundary_L(x, on_boundary):
    tol = 1E-24
    return on_boundary and near(x[0], 0, tol)
def boundary_R(x, on_boundary):
    tol = 1E-24
    return on_boundary and near(x[0], Lx, tol)

bc1 = DirichletBC(V.sub(0), 0, boundary_L)
bc2 = DirichletBC(V.sub(0), 1, boundary_R)
bcs = [bc1, bc2];


# Parameters
kappa = Constant(1);
#rlx_par_in = input("relaxation parameter? ")
rlx_par = Constant("0.1");#Constant(rlx_par_in);
#tol_abs_in = input("absolute tolerance? ")
tol_abs = Constant("1");#Constant(tol_abs_in);

# Define functions
dwr = TrialFunction(V)
(dw, dr) = split(dwr)
wr = interpolate( Expression(('x[0]<=0.5*Lx+DOLFIN_EPS ? -1 : 1', '1'), Lx=Lx, degree=2), V)

(w, r) = split(wr)

F = (-(1-w**4)*w**2*dw + (2/kappa**2)*w*inner(grad(w), grad(dw)) + 0.5*r*dw+ dr*(w**2-0.5) )*dx
#F = (-(1-w**2)*w*dw + (1/kappa**2)*inner(grad(w), grad(dw)) + 0.5*r*dw+ dr*(w-0.5) )*dx
solve(F == 0, wr, bcs,
   solver_parameters={"newton_solver":{"convergence_criterion":"residual","relaxation_parameter":rlx_par,"relative_tolerance":0.000001,"absolute_tolerance":tol_abs,"maximum_iterations":100}})


w = wr.sub(0, deepcopy=True)
r = wr.sub(1, deepcopy=True)

Constraint = assemble( (w**2-0.5)*dx)
print("Constraint violated by =", Constraint)

plot(w)
plt.title(r"$w(x)$",fontsize=26)
plt.show()
plot(w*w)
plt.title(r"$u(x)=w^2(x)$",fontsize=26)
plt.show()

