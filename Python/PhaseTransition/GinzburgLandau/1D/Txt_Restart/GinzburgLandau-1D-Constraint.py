#Here we solve the 1D Ginzbug Landau problem with an applied magnetic field.
#the whole formulation is presented in OneNote UH/superConductivity/Coding/1D Ginzburg Landau fenics.
#This is a modification of the original code in that we have added an integral constraint
#∫(u-1/2)=0.
#look specifically at sec-V
#The lagrange multiplier is r. 
#======================================================================================================
#The way the Code works
#1. The input to the code is:
#   a. The external field
#   b. The relaxation parameter
#   c. The absolute tolerance
#2. When reading from and writing into respective files,
#   we are writing the lagrange multiplier as a constant function
#   When reading the functions, we interpolate onto a space VAu.
#======================================================================================================
#ISSUES WITH THE CODE
#Cant get the input part to work. There is an issue combining functions read in from the input-Constraint.csv
#file. Best thing to do is to move on and work with "GinzburgLandau-1D-Constraint2.py". 


from dolfin import *
import fenics as fe
from ufl import tanh
import matplotlib.pyplot as plt
from ExtFile import write2Const, read4mConst, ExpressionFromScipyFunction, printout

# Optimization options for the form compiler
#parameters["form_compiler"]["cpp_optimize"] = True
parameters["krylov_solver"]["nonzero_initial_guess"] = True
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}

#Create mesh and define function space
Lx=500
mesh = fe.IntervalMesh(10000,0,Lx)
x = SpatialCoordinate(mesh)
VA = FiniteElement("CG", mesh.ufl_cell(), 2)
Vu = FiniteElement("CG", mesh.ufl_cell(), 2)
R = FiniteElement("Real", mesh.ufl_cell(), 0)
Vcoord = FunctionSpace(mesh, "Lagrange", 1)#This is for ExtFile
V = FunctionSpace(mesh, MixedElement(VA, Vu, R))
VAFnSp = FunctionSpace(mesh, "Lagrange", 2)
VuFnSp = FunctionSpace(mesh, "Lagrange", 2)
RFnSp = FunctionSpace(mesh, "Real", 0)
VAu = FunctionSpace(mesh, MixedElement(VA, Vu))#This is used to construct Au in ExtFile.py.

# Define functions
dAur = TrialFunction(V)
(dA, du, dr) = split(dAur)


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
rlx_par_in = input("relaxation parameter? ")
rlx_par = Constant(rlx_par_in);
tol_abs_in = input("absolute tolerance? ")
tol_abs = Constant(tol_abs_in);
#step_in = input("No. of Steps? ")
#step = Constant(step_in);
Ae = H*x[0]


#-----------------------------------------------------------------------------------------------------------------
#!!xDx!! ##!!xDx!! Newton rhapson Approach
#-----------------------------------------------------------------------------------------------------------------
#Compute first variation of Pi (directional derivative about u in the direction of v)
#Aur = interpolate( Expression(("1","0.0", "1.5"), degree=2), V)#SC phase as initial cond.
#Aur = interpolate( Expression(("H*x[0]","0", "0"), H=H, degree=2), V)#Normal phase as initial condiiton
#Coexistence of phase as initial condition
ul = Expression('0', degree=2, domain=mesh)
Al = Expression('H*(0.5*Lx-x[0])', H=H, Lx=Lx, degree=2, domain=mesh)
ur = Expression('1', degree=2, domain=mesh)
Ar = Expression('0', degree=2, domain=mesh)
Aur = interpolate( Expression(('x[0] <= 0.5*Lx + DOLFIN_EPS ? Al : Ar', 'x[0]<=0.5*Lx+DOLFIN_EPS ? ul : ur', '11'), ul=ul, ur=ur, Al=Al, Ar=Ar, Lx=Lx, degree=2), V)
#For 1D Vortex Solution.
#Aur = interpolate( Expression(("sqrt(2*tanh(x[0]+0.89)*tanh(x[0]+0.89)-1)","-sqrt(2)*sqrt(1-tanh(x[0]+0.89)*tanh(x[0]+0.89))","1"), degree=3), V)#1D vortex solution.
#---------------------------------------------------------------------------------------------------------------
##Reading input from a .csv file.
#Au0,r0 = read4mConst('input-Constraint.csv',VAu)
#(A0, u0) = split(Au0)
#Aur = Function(V)
##A = interpolate(A0, VAFnSp)
##u = interpolate(u0, VuFnSp)
##r = interpolate(r0,R)
#assigner = FunctionAssigner(V,[VAFnSp, VuFnSp, RFnSp])
#assigner.assign(Aur, [A0,u0,r0])
##Aur = interpolate( (A,u,r), V)
##Aur = interpolate( (Au,r), V)


(A, u, r) = split(Aur)


#In the following we scale the Energy by a constant (Scl) to make the residue tolerable when Lx becomes large
#print(Hin)
#if float(Hin)<=0.0:
# Scl = Constant(2/Lx)
#else:
# Scl = Constant(1/(float(Hin)*float(Hin)*Lx))
Scl=Constant(0.01);


F = Scl*(-(1-u**2)*u*du + (1/kappa**2)*inner(grad(u), grad(du)) + A**2*u*du + 0.5*r*du + dr*(u-0.5) + u**2*A*dA + inner(grad(A), grad(dA)))*dx + Scl*H*dA*ds
#solver.parameters.nonzero_initial_guess = True
solve(F == 0, Aur, bcs,
   #solver_parameters={"newton_solver":{"convergence_criterion":"residual","relaxation_parameter":0.01,"relative_tolerance":0.000001,"absolute_tolerance":0.001,"maximum_iterations":500}})
   solver_parameters={"newton_solver":{"convergence_criterion":"residual","relaxation_parameter":rlx_par,"relative_tolerance":0.000001,"absolute_tolerance":tol_abs,"maximum_iterations":10000}})

A = Aur.sub(0, deepcopy=True)
u = Aur.sub(1, deepcopy=True)
r = Aur.sub(2, deepcopy=True)

pie = assemble((1/(Lx))*((1-u**2)**2/2 + (1/kappa**2)*inner(grad(u), grad(u)) + A**2*u**2 + inner(grad(A-Ae), grad(A-Ae)))*dx )
print("Energy density =", pie)
Constraint = assemble( (u-0.5)*dx)
print("Constraint violated by =", Constraint)

##Save solution in a .csv file
write2Const(Vcoord,A,u,r,'output-Constraint.csv')

plot(u)
plt.title(r"$u(x)$",fontsize=26)
plt.show()
plot(A)
plt.title(r"$A(x)e_2$",fontsize=26)
plt.show()

