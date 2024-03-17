from dolfin import *
import fenics as fe
import numpy as np
import matplotlib.pyplot as plt

#Create mesh and define function space
Lx=500
mesh = fe.IntervalMesh(10000,0,Lx)
x = SpatialCoordinate(mesh)
VA = FiniteElement("CG", mesh.ufl_cell(), 2)
Vu = FiniteElement("CG", mesh.ufl_cell(), 2)
R = FiniteElement("Real", mesh.ufl_cell(), 0)
V = FunctionSpace(mesh, MixedElement(VA, Vu, R))
RFnSp = FunctionSpace(mesh, "Real", 0)
Vcoord = FunctionSpace(mesh, "Lagrange", 2)#This is for ExtFile

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
Ae = H*x[0]


#Newton Rhapson Initial guess
#-----------------------------------------------------------------------------------------------------------------
##Coexistence of phase as initial condition
#ul = Expression('0', degree=2, domain=mesh)
#Al = Expression('H*(0.5*Lx-x[0])', H=H, Lx=Lx, degree=2, domain=mesh)
#ur = Expression('1', degree=2, domain=mesh)
#Ar = Expression('0', degree=2, domain=mesh)
#Aur = interpolate( Expression(('x[0] <= 0.5*Lx + DOLFIN_EPS ? Al : Ar', 'x[0]<=0.5*Lx+DOLFIN_EPS ? ul : ur', '1'), ul=ul, ur=ur, Al=Al, Ar=Ar, Lx=Lx, degree=2), V)
#---------------------------------------------------------------------------------------------------------------
#Reading input from a .xdmf file.
Aur = Function(V)
A = Function(Vcoord)
u = Function(Vcoord)
r = Function(RFnSp)
data = np.loadtxt('test-2.txt')
y0 = data
r = interpolate(Constant(float(y0)),RFnSp)
A_in =  XDMFFile("test-0-Constraint.xdmf")
A_in.read_checkpoint(A,"A",0)
u_in =  XDMFFile("test-1-Constraint.xdmf")
u_in.read_checkpoint(u,"u",0)
assign(Aur,[A,u,r])


(A, u, r) = split(Aur)


Scl=Constant(1.000);


F = Scl*(-(1-u**2)*u*du + (1/kappa**2)*inner(grad(u), grad(du)) + A**2*u*du + 0.5*r*du + dr*(u-0.5) + u**2*A*dA + inner(grad(A), grad(dA)))*dx + Scl*H*dA*ds
solve(F == 0, Aur, bcs,
   solver_parameters={"newton_solver":{"convergence_criterion":"residual","relaxation_parameter":rlx_par,"relative_tolerance":0.000001,"absolute_tolerance":tol_abs,"maximum_iterations":500}})

A = Aur.sub(0, deepcopy=True)
u = Aur.sub(1, deepcopy=True)
r = Aur.sub(2, deepcopy=True)


##Save solution in a .xdmf file
Aur_split = Aur.split(True)
Aur_out = XDMFFile('test-0-Constraint.xdmf')
Aur_out.write_checkpoint(Aur_split[0], "A", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
Aur_out.close()
Aur_out = XDMFFile('test-1-Constraint.xdmf')
Aur_out.write_checkpoint(Aur_split[1], "u", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
Aur_out.close()
with open("test-2.txt", "w") as file:
    print(float(Aur_split[2]), file=file)

Constraint = assemble( (u-0.5)*dx)
print("Constraint violated by =", Constraint)


