import dolfin
print(f"DOLFIN version: {dolfin.__version__}")
from dolfin import *
import fenics as fe
import numpy as np
import ufl
print(f" UFL version: {ufl.__version__}")
from ufl import tanh
import matplotlib.pyplot as plt

# Parameters
lx = float(1.0)
ly = float(1.0)
Nx = int(40)
Ny = int(40)
# c_r = float(0.1)
# Ref_No = int(input("Refinement number? -->"))


# Create mesh and define function space
mesh = RectangleMesh(Point(0.0, 0.0), Point(lx, ly), Nx, Ny)
x = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, "Lagrange", 1)

##Reading in function
T = interpolate(
    Expression(
        "atan2(-0.5*(x[0]-0.5*lx)+rt*0.5*(x[1]-0.5*ly), 0.5*(x[1]-0.5*ly)+rt*0.5*(x[0]-0.5*lx) )",rt=np.sqrt(3),lx=lx,ly=ly,degree=1,),V,)

###---------------------------------------------------------------------------------------------------------------
Tarray = T.vector()[:]

try:
    import ufl
except ModuleNotFoundError:
    import ufl_legacy as ufl

uh = project(ufl.conditional(ufl.lt(T, 0), 1, 0), V)
print(uh.vector().get_local())


print("location of 0's", np.where(np.array(Tarray) == 0))
print("location of 2*pi's", np.where(np.array(Tarray) == 2 * np.pi))

c = plot(T)
plt.title(r"$\theta$(x)",fontsize=26)
plt.colorbar(c)
plt.show()
