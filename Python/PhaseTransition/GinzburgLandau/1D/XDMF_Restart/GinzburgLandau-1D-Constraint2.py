#Here we solve the 1D Ginzbug Landau problem with an applied magnetic field.
# the whole formulation is presented in OneNote UH/superConductivity/Coding/1D Ginzburg Landau fenics.
#This is a modification of the original code in that we have added an integral constraint
#\lambda(∫u-L/2)^2 with \lambda being a large number.
#look specifically at sec-VI
#======================================================================================================
#The Issues with the code
#1. IT starts jumping to a higher value of residue for the next trial. It is clearly unstable.

from dolfin import *
import fenics as fe
import matplotlib.pyplot as plt
#from ExtFile import write2, read4m, ExpressionFromScipyFunction, printout
import numpy as np

# Optimization options for the form compiler
parameters["krylov_solver"]["nonzero_initial_guess"] = True
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}


class ExpressionFromScipyFunction(Expression):
 def __init__(self, f, *args, **kwargs):
  self._f = f
  UserExpression.__init__(self, **kwargs)
 def eval(self, values, x):
  values[:] = self._f(*x)


#Create mesh and define function space
Lx=500
pord=1
mesh = fe.IntervalMesh(10000,0,Lx)
x = SpatialCoordinate(mesh)
VA = FiniteElement("CG", mesh.ufl_cell(), pord)
Vu = FiniteElement("CG", mesh.ufl_cell(), pord)
Vcoord = FunctionSpace(mesh, "Lagrange", 1)#We use this for read & write using ExtFile.py
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
#Hin = input("External Magnetic field? ")
#H = Constant(Hin);
H = Constant(0.707);
Hin = Constant(0.707);
rlx_par_in = input("relaxation parameter? ")
rlx_par = Constant(rlx_par_in);
tol_abs_in = input("absolute tolerance? ")
tol_abs = Constant(tol_abs_in);
Ae = H*x[0]
lmbda = Constant(1000000000);


#-----------------------------------------------------------------------------------------------------------------
#!!xDx!! ##!!xDx!! Newton rhapson Approach
#-----------------------------------------------------------------------------------------------------------------
#Compute first variation of Pi (directional derivative about u in the direction of v)
#Au = interpolate( Expression(("1","0.0", "1.5"), degree=2), V)#SC phase as initial cond.
#Au = interpolate( Expression(("H*x[0]","0", "0"), H=H, degree=2), V)#Normal phase as initial condiiton
#Au = interpolate( Expression(("0.5*h*x[0]","0.5*tanh(x[0]-0.5*l)+0.5"), l=Lx, h=H,  degree=1), V)#coexistence of phase as initial cond.
#Coexistence of phase as initial condition
ul = Expression('0', degree=2, domain=mesh)
Al = Expression('H*(0.5*Lx-x[0])', H=H, Lx=Lx, degree=2, domain=mesh)
ur = Expression('1', degree=2, domain=mesh)
Ar = Expression('0', degree=2, domain=mesh)
Au = interpolate( Expression(('x[0] <= 0.5*Lx + DOLFIN_EPS ? Al : Ar', 'x[0]<=0.5*Lx+DOLFIN_EPS ? ul : ur'), ul=ul, ur=ur, Al=Al, Ar=Ar, Lx=Lx, degree=2), V)
#---------------------------------------------------------------------------------------------------------------
##Reading input from a .xdmf file.
#Au = Function(V)
#A = Function(Vcoord)
#u = Function(Vcoord)
#A_in =  XDMFFile("test-0-Constraint2.xdmf")
#A_in.read_checkpoint(A,"A",0)
#u_in =  XDMFFile("test-1-Constraint2.xdmf")
#u_in.read_checkpoint(u,"u",0)
#assign(Au,[A,u])


(A, u) = split(Au)

Scl=1/100000;

r = assemble((u-0.5)*dx)
F = Scl*(-(1-u**2)*u*du + (1/kappa**2)*inner(grad(u), grad(du)) + A**2*u*du + r*lmbda*du + u**2*A*dA + inner(grad(A), grad(dA)))*dx + Scl*H*dA*ds

solve(F == 0, Au, bcs,
   solver_parameters={"newton_solver":{"convergence_criterion":"residual","relaxation_parameter":rlx_par,"relative_tolerance":0.000001,"absolute_tolerance":tol_abs,"maximum_iterations":500}})

A = Au.sub(0, deepcopy=True)
u = Au.sub(1, deepcopy=True)

##Save solution in a .xdmf file
Au_split = Au.split(True)
Au_out = XDMFFile('test-0-Constraint2.xdmf')
Au_out.write_checkpoint(Au_split[0], "A", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
Au_out.close()
Au_out = XDMFFile('test-1-Constraint2.xdmf')
Au_out.write_checkpoint(Aur_split[1], "u", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
Au_out.close()


pie = assemble((1/(Lx))*((1-u**2)**2/2 + (1/kappa**2)*inner(grad(u), grad(u)) + A**2*u**2 + inner(grad(A-Ae), grad(A-Ae)))*dx )
print("Energy density =", pie)
Constraint = assemble( (u-0.5)*dx)
print("Constraint violated by =", Constraint)

plot(u)
plt.title(r"$u(x)$",fontsize=26)
plt.show()
plot(A)
plt.title(r"$A(x)e_2$",fontsize=26)
plt.show()

