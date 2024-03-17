#The objective of this code is to get an understanding of "Coefficient".
# Most if it is taken from "Books/Computational_Work/Python/FEniCS/UFL Documentation.pdf" (For UFL 2021)
# or "https://fenics.readthedocs.io/projects/ufl/en/2019.1.0/manual/form_language.html#ad" (For Ufl 2019)


import dolfin
print(f"DOLFIN version: {dolfin.__version__}")
from dolfin import *
import ufl
print(f" UFL version: {ufl.__version__}")
import matplotlib.pyplot as plt
from petsc4py import PETSc


L=1
mesh = IntervalMesh(100,0,L)
x = SpatialCoordinate(mesh)
#element = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
V = FunctionSpace(mesh, "CG", 1)

#This is from p 31/231 of "Books/Computational_Work/Python/FEniCS/UFL Documentation.pdf" highlighted in pink.
#v = TestFunction(V)
#u = TrialFunction(V)
#w = ufl.Coefficient(V)
#
#f = inner(grad(w), grad(w))/2 * dx #w**2/2*dx
#F = derivative(f, w, v)
#J = derivative(F, w, u)


#This is from p 32 of "Books/Computational_Work/Python/FEniCS/UFL Documentation.pdf"
element = FiniteElement("CG", mesh.ufl_cell(), 1) #cell |--> mesh.ufl_cell()
w = ufl.Coefficient(element)
f = w**4/4*dx + inner(grad(w), grad(w))*dx #dx(i) |--> dx
F = ufl.derivative(f, w)
J = ufl.derivative(F, w)
Ja = action(J, w)
Jp = ufl.adjoint(J)
Jpa = action(Jp, w)
g = ufl.Coefficient(element)
Jnorm = ufl.energy_norm(J, g)







