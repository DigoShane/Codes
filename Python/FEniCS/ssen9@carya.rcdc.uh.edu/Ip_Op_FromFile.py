#The idea of this code is to write a small i/p-/op routine. I want to read from one file and write to another.
#The idea of this code is to write a small i/p-/op routine. I want to read from one file and write to another.
#I will achieve this using the simple Poisson equation example. 
#I will read f from a file "foo.csv" and write u to "foo1.csv"
#The equation to solve is:-
#Ω=[0,1]^2, \Gamma_D={(0,y)U(1,y)} and \Gamma_N = ∂Ω\ \Gamma_D
#-∆u=f* in Ω
# u =0 on \Gamma_D
# ∂u/∂n =g on \Gamma_N
#f= L(f* )
#g= sin(5x)
#----------------------------------------------------------------
#where f* is defined as
#-∆f*=f in Ω
# f* =0 on \Gamma_D
# ∂f*/∂n =g on \Gamma_N
#f= 10e^(-20(x-0.5)^2-20(y-0.5)^2)



from fenics import *
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.interpolate import interp1d, interp2d

class ExpressionFromScipyFunction(Expression):
 def __init__(self, f, *args, **kwargs):
  self._f = f
  UserExpression.__init__(self, **kwargs)
 def eval(self, values, x):
  values[:] = self._f(*x)



# Create mesh and define function space
mesh = UnitSquareMesh(32, 32)
V = FunctionSpace(mesh, "Lagrange", 1)

# Define Dirichlet boundary (x = 0 or x = 1)
def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)

# Defining the load
##method1
#f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2)
#method2
data2 = np.loadtxt('foo.csv')
y0, y1, values2 = data2[:,0], data2[:,1], data2[:,2]
interpolant2 = interp2d(y0, y1, values2, kind='linear', copy=False, bounds_error=True)
expression2 = ExpressionFromScipyFunction(interpolant2, element=V.ufl_element())
f = interpolate(expression2, V)
plot(f)
plt.show()

# Defining g, a and L
g = Expression("sin(5*x[0])", degree=2)
a = inner(grad(u), grad(v))*dx
L = f*v*dx + g*v*ds

# Compute solution
u = Function(V)
solve(a == L, u, bc)

#Save solution in a .csv file
coords = V.tabulate_dof_coordinates()
vec = u.vector().get_local()
outfile = open("foo1.csv", "w")
for coord, val in zip(coords, vec):
    print(coord[0], coord[1], val, file=outfile)
    #print("----------xxxxx---------")
    #print(coord[0], coord[1], val)
    #print("------------------------")
    #print(coord[0], coord[1])
    #print("------------------------")

# Plot solution
plot(u)
plt.show()


