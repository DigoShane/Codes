#Here we solve the 1D Ginzbug Landau problem with an applied magnetic field.
# the whole formulation is presented in OneNote UH/superConductivity/Coding/1D Ginzburg Landau fenics.
#This is a modification of the original code in that we have added an integral constraint
#\lambda(∫u-L/2)^2 with \lambda being a large number.
#look specifically at sec-VI

from dolfin import *
import fenics as fe
import matplotlib.pyplot as plt
#from ExtFile import write2, read4m, ExpressionFromScipyFunction, printout
import numpy as np
import scipy as sp
from scipy.interpolate import interp1d, interp2d

# Optimization options for the form compiler
#parameters["form_compiler"]["cpp_optimize"] = True
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
Lx=10
pord=1
mesh = fe.IntervalMesh(200,0,Lx)
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
##Reading input from a .csv file.
#data2 = np.loadtxt('input.csv')
#y0, values1, values2 = data2[:,0], data2[:,1], data2[:,2]
#values = np.transpose( np.column_stack((values1, values2)) )
#print(y0)
#print("----------xxxxx---------")
#print(values1)
#print("----------xxxxx---------")
#print(values2)
#print("----------xxxxx---------")
#print(values)
#print("----------xxxxx---------")
#interpolant2 = interp1d(y0, values, kind='linear', copy=False, bounds_error=True)
#expression2 = ExpressionFromScipyFunction(interpolant2, element=V.ufl_element())
#Au = interpolate(expression2, V)

(A, u) = split(Au)

Scl=1/10000000;

r = assemble((u-0.5)*dx)
F = Scl*(-(1-u**2)*u*du + (1/kappa**2)*inner(grad(u), grad(du)) + A**2*u*du + r*lmbda*du + u**2*A*dA + inner(grad(A), grad(dA)))*dx + Scl*H*dA*ds
#solver.parameters.nonzero_initial_guess = True
solve(F == 0, Au, bcs,
   #solver_parameters={"newton_solver":{"convergence_criterion":"incremental","relaxation_parameter":0.01,"relative_tolerance":0.000001,"absolute_tolerance":0.001,"maximum_iterations":500}})
   solver_parameters={"newton_solver":{"convergence_criterion":"residual","relaxation_parameter":rlx_par,"relative_tolerance":0.000001,"absolute_tolerance":tol_abs,"maximum_iterations":5000}})

A = Au.sub(0, deepcopy=True)
u = Au.sub(1, deepcopy=True)

##Save solution in a .csv file
coords = Vcoord.tabulate_dof_coordinates()
#print(coords[:,0])
#print("----------------------------------------")
vecA = A.vector().get_local()
#print(vecA)
#print("----------------------------------------")
vecu = u.vector().get_local()
#print(vecu)
#print("----------------------------------------")
with open("output.csv","w") as outfile:
 for coord, val1, val2 in zip(coords, vecA, vecu):
    print('{:16.8f}'.format(coord[0]), '{:16.8f}'.format(val1), '{:16.8f}'.format(val2), file=outfile)#(x,A,u)
    #print("----------xxxxx---------")
    #print('{:16.8f}'.format(coord[0]), '{:16.8f}'.format(val1), '{:16.8f}'.format(val2))
outfile.closed


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

