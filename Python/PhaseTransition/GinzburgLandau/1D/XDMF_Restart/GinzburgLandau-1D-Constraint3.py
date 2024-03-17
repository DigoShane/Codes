#Here we solve the 1D Ginzbug Landau problem with an applied critical field, H=1/√2.
# the whole formulation is presented in OneNote UH/superConductivity/Coding/Ginsburg landau square argument
#This is a modification of the original code in that we have added an integral constraint and used w^2=u to ensure u≥0.
#We got rid of the extra solution associated with the above formulation.
#look specifically at "sec-1D weak Form"
#======================================================================================================
#The Issues with the code
#1. There seems to be a multiple root in this problem. If i use the old problem, the algorithm works.
#   However, for the new problem, it blows up suggesting that the jacobian was singular.
#------------------------------------------------------------------------------------------------------
#ISSUES WITH THE CODE
#1. The code doesnt work cause the Jacobian seems to blow up. This is troublesome since we have u=0 as our desired solution.
#   I think thats where the jacobian blows up.


from dolfin import *
import fenics as fe
import numpy as np
from ufl import tanh
import matplotlib.pyplot as plt

# Optimization options for the form compiler
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
Vw = FiniteElement("CG", mesh.ufl_cell(), 2)
R = FiniteElement("Real", mesh.ufl_cell(), 0)
V = FunctionSpace(mesh, MixedElement(VA, Vw, R))
RFnSp = FunctionSpace(mesh, "Real", 0)
Vcoord = FunctionSpace(mesh, "Lagrange", 2)#This is for ExtFile

# Define functions
dAwr = TrialFunction(V)
(dA, dw, dr) = split(dAwr)


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
Ae = H*x[0]


#-----------------------------------------------------------------------------------------------------------------
#!!xDx!! ##!!xDx!! Newton rhapson Approach
#-----------------------------------------------------------------------------------------------------------------
Awr = interpolate( Expression(("1","0.0", "1.5"), degree=2), V)#SC phase as initial cond.
#Awr = interpolate( Expression(("H*x[0]","0", "0"), H=H, degree=2), V)#Normal phase as initial condiiton
##Coexistence of phase as initial condition
#wl = Expression('0', degree=2, domain=mesh)
#Al = Expression('H*(0.5*Lx-x[0])', H=H, Lx=Lx, degree=2, domain=mesh)
#wr = Expression('1', degree=2, domain=mesh)
#Ar = Expression('0', degree=2, domain=mesh)
#Awr = interpolate( Expression(('x[0] <= 0.5*Lx + DOLFIN_EPS ? Al : Ar', 'x[0]<=0.5*Lx+DOLFIN_EPS ? wl : wr', '11'), wl=wl, wr=wr, Al=Al, Ar=Ar, Lx=Lx, degree=2), V)
#For 1D Vortex Solution.
#Awr = interpolate( Expression(("sqrt(2*tanh(x[0]+0.89)*tanh(x[0]+0.89)-1)","-sqrt(2)*sqrt(1-tanh(x[0]+0.89)*tanh(x[0]+0.89))","1"), degree=3), V)#1D vortex solution.
#---------------------------------------------------------------------------------------------------------------
##Reading input from a .xdmf file.
#Awr = Function(V)
#A = Function(Vcoord)
#w = Function(Vcoord)
#r = Function(RFnSp)
#data = np.loadtxt('test-2-Constraint3.txt')
#y0 = data
#r = interpolate(Constant(float(y0)),RFnSp)
#A_in =  XDMFFile("test-0-Constraint3.xdmf")
#A_in.read_checkpoint(A,"A",0)
#w_in =  XDMFFile("test-1-Constraint3.xdmf")
#w_in.read_checkpoint(w,"w",0)
#assign(Aur,[A,w,r])
##plot(w)
##plt.title(r"$w(x)-b4$",fontsize=26)
##plt.show()
##plot(A)
##plt.title(r"$A(x)e_2-b4$",fontsize=26)
##plt.show()

(A, w, r) = split(Awr)

plot(w)
plt.title(r"$w(x)-b4$",fontsize=26)
plt.show()
plot(A)
plt.title(r"$A(x)e_2-b4$",fontsize=26)
plt.show()



F = (-(1-w**4)*w**2*dw + A**2*w**2*dw + 0.5*r*dw + dr*(w**2-0.5) + w**4*A*dA )*dx + H*dA*ds # Original problem.
#F = (-(1-w**2)*w*dw + (1/kappa**2)*inner(grad(w), grad(dw)) + A**2*w*dw + 0.5*r*dw + dr*(w-0.5) + w**2*A*dA + inner(grad(A), grad(dA)))*dx + H*dA*ds#testing with original problem.
#F = (-(1-w**4)*w**2*dw + (2/kappa**2)*w*inner(grad(w), grad(dw)) + A**2*w**2*dw + 0.5*r*dw + dr*(w**2-0.5) + w**4*A*dA + inner(grad(A), grad(dA)))*dx + H*dA*ds # Original problem.
solve(F == 0, Awr, bcs,
   #solver_parameters={"newton_solver":{"convergence_criterion":"residual","relaxation_parameter":0.01,"relative_tolerance":0.000001,"absolute_tolerance":0.001,"maximum_iterations":500}})
   solver_parameters={"newton_solver":{"convergence_criterion":"residual","relaxation_parameter":rlx_par,"relative_tolerance":0.000001,"absolute_tolerance":tol_abs,"maximum_iterations":100}})

A = Aur.sub(0, deepcopy=True)
w = Aur.sub(1, deepcopy=True)
r = Aur.sub(2, deepcopy=True)

print("hello")

##Save solution in a .xdmf file
Awr_split = Awr.split(True)
Awr_out = XDMFFile('test-0-Constraint3.xdmf')
Awr_out.write_checkpoint(Awr_split[0], "A", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
Awr_out.close()
Awr_out = XDMFFile('test-1-Constraint3.xdmf')
Awr_out.write_checkpoint(Awr_split[1], "w", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
Awr_out.close()
with open("test-2-Constraint3.txt", "w") as file:
    print(float(Aur_split[2]), file=file)


pie = assemble((1/(Lx))*((1-w**4)**2/2 + (4/kappa**2)*w**2*inner(grad(w), grad(w)) + A**2*w**4 + inner(grad(A-Ae), grad(A-Ae)))*dx )
print("Energy density =", pie)
Constraint = assemble( (w**2-0.5)*dx)
print("Constraint violated by =", Constraint)


plot(w)
plt.title(r"$w(x)$",fontsize=26)
plt.show()
plot(w*w)
plt.title(r"$u(x)=w^2(x)$",fontsize=26)
plt.show()
plot(A)
plt.title(r"$A(x)e_2$",fontsize=26)
plt.show()

