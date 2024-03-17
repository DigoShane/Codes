#This is a code to solve the heat equation in 2D.
#∂u/∂t = ∆u + f in Ωx[0,T]
#u = u_D        on ∂Ωx[0,T]
#u = u_0        at t=0
#where f =ß - 2-2\alpha
#u_D = 1 + x^2 +\alpha y^2 +ßt
#u_0 = 1 + x^2 + \alpha y^2

from fenics import *
import numpy as np
import matplotlib.pyplot as plt

T = 2.0            # final time
num_steps = 10     # number of time steps
dt = T / num_steps # time step size
alpha = 3          # parameter alpha
beta = 1.2         # parameter beta

# Create mesh and define function space
nx = ny = 8
mesh = UnitSquareMesh(nx, ny)
V = FunctionSpace(mesh, 'P', 1)

# Define boundary condition
u_D = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t',
                 degree=2, alpha=alpha, beta=beta, t=0)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

# Define initial value
u_n = interpolate(u_D, V)
#u_n = project(u_D, V)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(beta - 2 - 2*alpha)

#F = u*v*dx + dt*dot(grad(u), grad(v))*dx - (u_n + dt*f)*v*dx
#a, L = lhs(F), rhs(F)
a = u*v*dx + dt*dot(grad(u), grad(v))*dx 
L =  (u_n + dt*f)*v*dx

# Time-stepping
u = Function(V)
t = 0
for n in range(num_steps):

    # Update current time
    t += dt
    u_D.t = t

    # Compute solution
    solve(a == L, u, bc)

    # Plot solution
    plot(u)

    # Compute error at vertices
    u_e = interpolate(u_D, V)
    error = np.abs(u_e.vector().get_local() - u.vector().get_local()).max()#https://fenicsproject.discourse.group/t/question-about-solution-vector-and-solution-array/2078
    print('t = %.2f: error = %.3g' % (t, error))

    # Update previous solution
    u_n.assign(u)

# Hold plot
plt.show()#https://stackoverflow.com/questions/53730427/fenics-did-not-show-figure-nameerror-name-interactive-is-not-defined
