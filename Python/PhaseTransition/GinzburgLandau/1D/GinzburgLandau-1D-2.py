#Here we solve the 1D Ginzbug Landau problem with an applied magnetic field.
# the whole formulation is presented in OneNote UH/superConductivity/Coding/1D Ginzburg Landau fenics.
#The equations that we solve for here are formulated differently. Here we do not set H'=0
#The energy minimization is given by
# ∫(1-u^2)^2/2+(u'/k)^2+A^2u^2 +(A'-H)^2
#where k:=\kappa
#The variation of the energy can be written as
#∫(1-u^2)(-2u∂u) + 2u'(∂u)'/k^2 + 2A(∂A)u^2 + 2u(∂u)A^2 + 2(A'-H)(∂A)'
#∫2∂u[-u(1-u^2) - u''/k^2 +A^2u] + (∂u)u'/k^2|_0^L + ∫2∂A[-(A'-H)'+Au^2] + (A'-H)|_0^L∂A|_0^L
#we have the boundary conditions u(0) = 0, u(L) = 1, A'(0) = H, A(L) = 0.
#where ∂u and ∂A are the variations.
#The PDE are:-
#-u(1-u^2) + A^2u=u''/k^2 in [0,Lx]
#(A'-H)'=Au^2
#u(0) = 0
#u(L) = 1
#A'(0) = H
#A(L) = 0
#----------------------------------------------------------------------------------------------------------
#If we want to reeconstructing the weak form of the above equations, give us
#∫-u(1-u^2)v-u''v/k^2+A^2uv=0
#∫-uv(1-u^2)+u'v'/k^2+A^2uv=0 
#where we used the fac thtat v(0)=v(L)=0
#∫-(A'-H)'B+ABu^2=0
#∫(A'-H)B'+ABu^2 - (A'-H)B|_0^L=0
#∫(A'-H)B'+ABu^2 =0
#where i have used B(L)=0 and A'(0)-H=0.
#Notice that the weak form is a bit different
#∫-uv(1-u^2)+u'v'/k^2+A^2uv+(A'-H)B'+ABu^2 = 0
#==========================================================================================================
# SOMETHING SEEMS TO BE WRONG HERE. 
#This matches the original code if we plug in negative values.
#While the shapes look similar, need to ensure that the output of the two codes are the same.
#   I have verified it for the case when H=-0.5 (0.5 for the original code).
#      ****it turns out there was a sign issue in the original code, ****
#      **** a plus should have been a minus in the weak form.        ****
#      **** Once fixed, the results matched much better.             *****



from dolfin import *
import fenics as fe
import matplotlib.pyplot as plt

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}

# Create mesh and define function space
Lx=150
mesh = fe.IntervalMesh(500,0,Lx)
x = SpatialCoordinate(mesh)
VA = FiniteElement("CG", mesh.ufl_cell(), 2)
Vu = FiniteElement("CG", mesh.ufl_cell(), 2)
V = FunctionSpace(mesh, MixedElement(VA, Vu))

# Define functions
#Bv = TestFunction(V)
#(B, v) = split(Bv)
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
kappa = Constant(2);
Hin = input("External Magnetic field? ")
H = Constant(Hin);
Ae = H*x[0]


#-----------------------------------------------------------------------------------------------------------------
#!!xDx!! ##!!xDx!! Newton rhapson Approach
#-----------------------------------------------------------------------------------------------------------------
#Compute first variation of Pi (directional derivative about u in the direction of v)
#Au = Function(V)
Au = interpolate( Expression(("0","1.5"), degree=2), V)#SC phase as initial cond.
#Au = interpolate( Expression(("H*x[0]","0"), H=H, degree=2), V)#Normal phase as initial condiiton
#Au = interpolate( Expression(("0.5*h*x[0]","0.5*tanh(x[0]-0.5*l)+0.5"), l=Lx, h=H,  degree=10), V)#coexistence of phase as initial cond.
(A, u) = split(Au)


F = (-(1-u**2)*u*du + (1/kappa**2)*inner(grad(u), grad(du)) + A**2*u*du + u**2*A*dA + inner(grad(A-Ae), grad(dA)))*dx 
#solver.parameters.nonzero_initial_guess = True
solve(F == 0, Au, bcs,
   solver_parameters={"newton_solver":{"convergence_criterion":"incremental","relaxation_parameter":0.01,"relative_tolerance":0.000001,"absolute_tolerance":0.001,"maximum_iterations":500}})

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


