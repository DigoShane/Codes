import numpy as np
import dolfin
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

mesh = dolfin.UnitIntervalMesh(3)
V = dolfin.FunctionSpace(mesh, "Lagrange", 4)
u = dolfin.Function(V)
u.interpolate(dolfin.Expression("x[0]", degree=1))


x = V.tabulate_dof_coordinates()
vals0 = u.vector().get_local()
t0 = 0.2

print(x)

u.interpolate(dolfin.Expression("2*x[0]*x[0]", degree=2))
vals1 = u.vector().get_local()
t1 = 0.3

X = np.tile(x.reshape(-1), 2).reshape(2, -1)
Y = np.zeros((2, x.shape[0]))
Y[0] = t0
Y[1] = t1

print(X)
print(Y)

Z = np.vstack([vals0, vals1])
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, Z)
#surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                       linewidth=0, antialiased=False)

plt.xlabel("x")
plt.ylabel("t")
plt.show()
