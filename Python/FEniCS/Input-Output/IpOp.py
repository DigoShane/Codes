from dolfin import *
import numpy as np
from scipy.interpolate import interp1d, interp2d
import matplotlib.pyplot as plt


class ExpressionFromScipyFunction(Expression):
 def __init__(self, f, *args, **kwargs):
  self._f = f
  UserExpression.__init__(self, **kwargs)
 def eval(self, values, x):
  values[:] = self._f(*x)

mesh1 = UnitIntervalMesh(4)
V1 = FunctionSpace(mesh1, 'CG', 1)

data1 = np.loadtxt('Data1.csv')
x, values1 = data1[:,0], data1[:,1]
interpolant1 = interp1d(x, values1, kind='linear', copy=False, bounds_error=True)
expression1 = ExpressionFromScipyFunction(interpolant1, element=V1.ufl_element())
u1 = interpolate(expression1, V1)

plot(u1)
plt.show()

#======================================================================================

data2 = np.loadtxt('Data2.csv')
y0, y1, values2 = data2[:,0], data2[:,1], data2[:,2]
interpolant2 = interp2d(y0, y1, values2, kind='linear', copy=False, bounds_error=True)

mesh2 = UnitSquareMesh(3, 3)
V2 = FunctionSpace(mesh2, 'CG', 1)

expression2 = ExpressionFromScipyFunction(interpolant2, element=V2.ufl_element())
u2 = interpolate(expression2, V2)

plot(u2)
plt.show()

#======================================================================================

data3 = np.loadtxt('Data3.csv')
y0, values31, values32 = data3[:,0], data3[:,1], data3[:,2]
values3 = np.transpose( np.column_stack((values31, values32)) )
print( values3 )
print( y0 )
interpolant3 = interp1d(y0, values3, kind='linear', copy=False, bounds_error=True)

mesh3 = UnitIntervalMesh(4)
V31 = FiniteElement("CG", mesh3.ufl_cell(), 1)
V32 = FiniteElement("CG", mesh3.ufl_cell(), 1)
V3 = FunctionSpace(mesh3, MixedElement(V31, V32))

expression3 = ExpressionFromScipyFunction(interpolant3, element=V3.ufl_element())
u3 = interpolate(expression3, V3)

u31 = u3.sub(0, deepcopy=True)
u32 = u3.sub(1, deepcopy=True)

plot(u31)
plot(u32)
plt.show()


