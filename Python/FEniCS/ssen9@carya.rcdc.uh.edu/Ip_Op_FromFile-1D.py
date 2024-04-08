#I will achieve this using the simple Poisson equation example. 
#I will read f from a file "1Dfoo.csv" and write u to "1Dfoo1.csv"
#The output of method 1 is stored in "1Dfoo1.csv" and then we change
#the name to "1Dfoo.csv" and run method 2.
#The equation to solve is:-
#Ω=[0,1]
#-∆u=f* in Ω
#u =0 on ∂Ω
#f= L(f* )
#----------------------------------------------------------------
#where f* is defined as
#-∆f*=f in Ω
# f* =0 on ∂Ω
#f= exp(-x^2)


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
mesh = UnitIntervalMesh(8)
pord=1;
V = FunctionSpace(mesh, "Lagrange", pord)

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
#f = Expression("exp(-x[0]*x[0])", degree=pord)
#method2
data2 = np.loadtxt('1Dfoo.csv')
y0, values2 = data2[:,0], data2[:,1]
#for coord, val in zip(y0, values2):
# print("----------xxxxx---------")
# print('{:16.8f}'.format(coord), '{:16.8f}'.format(val))
print(y0)
print("----------xxxxx---------")
print(values2)
print("----------xxxxx---------")
interpolant2 = interp1d(y0, values2, kind='linear', copy=False, bounds_error=True)
expression2 = ExpressionFromScipyFunction(interpolant2, element=V.ufl_element())
f = interpolate(expression2, V)
#plot(f)
#plt.show()

# Defining a and L
a = inner(grad(u), grad(v))*dx
L = f*v*dx

# Compute solution
u = Function(V)
solve(a == L, u, bc)

#Save solution in a .csv file
coords = V.tabulate_dof_coordinates()
vec = u.vector().get_local()
with open("1Dfoo1.csv","w") as outfile:
 for coord, val in zip(coords, vec):
  #print(coord[0], val, file=outfile)
  print('{:16.8f}'.format(coord[0]), '{:16.8f}'.format(val), file=outfile)
  print("----------xxxxx---------")
  print('{:16.8f}'.format(coord[0]), '{:16.8f}'.format(val))
outfile.closed






# Plot solution
plot(u)
plt.show()


