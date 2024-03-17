#this is based on the formulation presented in "https://comet-fenics.readthedocs.io/en/latest/demo/thermoelasticity/thermoelasticity_transient.html"
#The eqns and presented in the above url.
# The actual problem is a plate with a hole in the centre. The temperature at the hole's inner surface is higher than ambient; this is specified
#There is flux free boundary conditions on the outer surfaces. 
#Using the symmetry of the problem, we consider a quarter section of the problem. The symmetry conditions impose that the bottom face has u_y=0
#the left face has u_x=0. We can show \epsilon_xy=0 on the bottom and left surface. Physically, shear would destroy the symmetry.

from dolfin import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt
#matplotlib notebook


# !!xDx!!# Optimization options for the form compiler
# !!xDx!!parameters["form_compiler"]["cpp_optimize"] = True
# !!xDx!!ffc_options = {"optimize": True, \
# !!xDx!!               "eliminate_zeros": True, \
# !!xDx!!               "precompute_basis_const": True, \
# !!xDx!!               "precompute_ip_const": True}


L = 1.
R = 0.1
N = 64  # mesh density

domain = Rectangle(Point(0., 0.), Point(L, L)) - Circle(Point(0., 0.), R, 100)
mesh = generate_mesh(domain, N)

T0 = Constant(293.)
DThole = Constant(10.)
E = 70e3
nu = 0.3
lmbda = Constant(E*nu/((1+nu)*(1-2*nu)))
mu = Constant(E/2/(1+nu))
rho = Constant(2700.)     # density
alpha = 2.31e-5  # thermal expansion coefficient
kappa = Constant(alpha*(2*mu + 3*lmbda))
cV = Constant(910e-6)*rho  # specific heat per unit volume at constant strain
k = Constant(237e-6)  # thermal conductivity

Vue = VectorElement("CG", mesh.ufl_cell(), 2) # displacement finite element
Vte = FiniteElement("CG", mesh.ufl_cell(), 1) # temperature finite element
V = FunctionSpace(mesh, MixedElement(Vue, Vte))

def inner_boundary(x, on_boundary):
    return near(x[0]**2+x[1]**2, R**2, 1e-3) and on_boundary
def bottom(x, on_boundary):
    return near(x[1], 0) and on_boundary
def left(x, on_boundary):
    return near(x[0], 0) and on_boundary

bc1 = DirichletBC(V.sub(0).sub(1), Constant(0.), bottom)
bc2 = DirichletBC(V.sub(0).sub(0), Constant(0.), left)
bc3 = DirichletBC(V.sub(1), DThole, inner_boundary)
bcs = [bc1, bc2, bc3]

U_ = TestFunction(V)
(u_, Theta_) = split(U_)
dU = TrialFunction(V)
(du, dTheta) = split(dU)
Uold = Function(V)
(uold, Thetaold) = split(Uold)


def eps(v):
    return sym(grad(v))


def sigma(v, Theta):
    return (lmbda*tr(eps(v)) - kappa*Theta)*Identity(2) + 2*mu*eps(v)


dt = Constant(0.)
mech_form = inner(sigma(du, dTheta), eps(u_))*dx
therm_form = (cV*(dTheta-Thetaold)/dt*Theta_ +
              kappa*T0*tr(eps(du-uold))/dt*Theta_ +
              dot(k*grad(dTheta), grad(Theta_)))*dx
form = mech_form + therm_form

Nincr = 100
t = np.logspace(1, 4, Nincr+1)
Nx = 100
x = np.linspace(R, L, Nx)
T_res = np.zeros((Nx, Nincr+1))
U = Function(V)
for (i, dti) in enumerate(np.diff(t)):
    print("Increment " + str(i+1))
    dt.assign(dti)
    solve(lhs(form) == rhs(form), U, bcs)
    Uold.assign(U)
    T_res[:, i+1] = [U(xi, 0.)[2] for xi in x]

plt.figure()
plt.plot(x, T_res[:, 1::Nincr//10])
plt.xlabel("$x$-coordinate along $y=0$")
plt.ylabel("Temperature variation $\Theta$")
plt.legend(["$t={:.0f}$".format(ti) for ti in t[1::Nincr//10]], ncol=2)
plt.show()

