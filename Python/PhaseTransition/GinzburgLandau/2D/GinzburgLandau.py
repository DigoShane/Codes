#Here we solve the 2D Ginzbug Landau problem with an applied magnetic field.
#the whole formulation is presented in OneNote UH/superConductivity/Coding/2D Ginzburg Landau fenics.
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
#Things to keep in mind about writing this code:-
#1. Define a functoon to evaluate the curl
#2. Define a rotation funciton.
#3. HAve replace L with l throught.
#4. All variables are lower case.
#5. REdo the code by using Hn\cdot B\perp
#======================================================================================================
#ISSUES WITH THE CODE:-
#1. I keep getting a warning .....
#   " *** Warning: Found no facets matching domain for boundary condition. "
#    I thought it was wrt the hole in the domain (bcoz of paraview) but i dont think it is,
#    since i am getting the same error gorm the GinzburgLAndau-theta.py code which has no hole.
#    MY GUESS IS, IT HAS TO DO WITH THE WAY I DEFINE THE BOUNDARY CONDITIONS..... CHECK



from dolfin import *
import fenics as fe
import numpy as np
from ufl import tanh
import matplotlib.pyplot as plt
import mshr


#Create mesh and define function space
l = 10
r = 0.1
domain = mshr.Rectangle(Point(-l,-l), Point(l, l)) - mshr.Circle(Point(0., 0.), r)
mesh = mshr.generate_mesh(domain, int(l)*10)
x = SpatialCoordinate(mesh)
Va1 = FiniteElement("CG", mesh.ufl_cell(), 2)
Va2 = FiniteElement("CG", mesh.ufl_cell(), 2)
Vu = FiniteElement("CG", mesh.ufl_cell(), 2)
V = FunctionSpace(mesh, MixedElement(Va1, Va2, Vu))
Vcoord = FunctionSpace(mesh, "Lagrange", 2)#This is for ExtFile


#Boundary conditions
class Inner(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0]**2+x[1]**2, r, DOLFIN_EPS)

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1] , l , DOLFIN_EPS) 

class Left(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0] , -l , DOLFIN_EPS) 

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1] , -l , DOLFIN_EPS) 

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0] , l , DOLFIN_EPS) 


boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
iinner = Inner()
top = Top()
left = Left()
bottom = Bottom()
right = Right()

top.mark(boundaries, 2)
left.mark(boundaries, 3)
bottom.mark(boundaries, 4)
right.mark(boundaries, 5)
iinner.mark(boundaries, 1) 


bc1 = DirichletBC(V.sub(1), 1, boundaries, 2) #Setting u=1 on top boundary
bc2 = DirichletBC(V.sub(1), 1, boundaries, 3) #Setting u=1 on left boundary
bc3 = DirichletBC(V.sub(1), 1, boundaries, 4) #Setting u=1 on bottom boundary
bc4 = DirichletBC(V.sub(1), 1, boundaries, 5) #Setting u=1 on right boundary
bc5 = DirichletBC(V.sub(1), 0, boundaries, 1) #Setting u=0 on inner boundary
bc6 = DirichletBC(V.sub(0), 100, boundaries, 1)#A on the inner boundary, is 100
bcs = [bc1, bc2, bc3, bc4, bc5, bc6];

ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

File("boundaries.pvd").write(boundaries)

# Define functions
da1a2u = TrialFunction(V)
(da1, da2, du) = split(da1a2u)
#n = FacetNormal(mesh)



# Parameters
kappa = Constant(1);
Hin = input("External Magnetic field? ")
H = Constant(Hin);
rlx_par_in = input("relaxation parameter? ")
rlx_par = Constant(rlx_par_in);
tol_abs_in = input("absolute tolerance? ")
tol_abs = Constant(tol_abs_in);
Ae = H*x[0]


def curl(a1,a2):
    return a1.dx(0) - a2.dx(1)

#Newton rhapson Approach
a1a2u = interpolate( Expression(("0.0","0.0", "1"), degree=2), V)#SC phase as initial cond.
#---------------------------------------------------------------------------------------------------------------
##Reading input from a .xdmf file.
#a1a2u = Function(V)
#a1 = Function(Vcoord)
#a2 = Function(Vcoord)
#u = Function(RFnSp)
#a1_in =  XDMFFile("GL-0.xdmf")
#a1_in.read_checkpoint(a1,"a1",0)
#a2_in =  XDMFFile("GL-1.xdmf")
#a2_in.read_checkpoint(a2,"a2",0)
#u_in =  XDMFFile("GL-2.xdmf")
#u_in.read_checkpoint(u,"u",0)
#assign(a1a2u,[a1,a2,u])
##plot(u)
##plt.title(r"$u(x)-b4$",fontsize=26)
##plt.show()
##plot(A)
##plt.title(r"$A(x)e_2-b4$",fontsize=26)
##plt.show()

(a1, a2, u) = split(a1a2u)


F = (-(1-u**2)*u*du + (1/kappa**2)*inner(grad(u), grad(du)) + (a1**2+a2**2)*u*du + u**2*(a1*da1+a2*da2) + inner(curl(a1,a2), curl(da1,da2)))*dx + H*(-da2*ds(5)-da1*ds(4)+da2*ds(3)+da1*ds(2))
solve(F == 0, a1a2u, bcs,
   solver_parameters={"newton_solver":{"convergence_criterion":"residual","relaxation_parameter":rlx_par,"relative_tolerance":0.000001,"absolute_tolerance":tol_abs,"maximum_iterations":10}})

a1 = a1a2u.sub(0, deepcopy=True)
a2 = a1a2u.sub(1, deepcopy=True)
u = a1a2u.sub(2, deepcopy=True)


###Save solution in a .xdmf file
#a1a2u_split = a1a2u.split(True)
#a1a2u_out = XDMFFile('GL-0.xdmf')
#a1a2u_out.write_checkpoint(a1a2u_split[0], "a1", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
#a1a2u_out.close()
#a1a2u_out = XDMFFile('GL-1.xdmf')
#a1a2u_out.write_checkpoint(a1a2u_split[1], "a2", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
#a1a2u_out.close()
#a1a2u_out = XDMFFile('GL-2.xdmf')
#a1a2u_out.write_checkpoint(a1a2u_split[2], "u", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
#a1a2u_out.close()


pie = assemble((1/(4*l*l))*((1-u**2)**2/2 + (1/kappa**2)*inner(grad(u), grad(u)) + (a1**2+a2**2)*u**2 + inner(curl(a1 ,a2-Ae), curl(a1 ,a2-Ae) ))*dx )
#divide by volume, 4l^2-πr^2.
print("Energy density =", pie)


plot(u)
plt.title(r"$u(x)$",fontsize=26)
plt.show()
plot(a1)
plt.title(r"$A_1(x)$",fontsize=26)
plt.show()
plot(a2)
plt.title(r"$A_2(x)$",fontsize=26)
plt.show()



