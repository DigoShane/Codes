#The objective of this code is to decide the node numberings after refinement.

from fenics import *
import matplotlib.pyplot as plt


mesh = UnitSquareMesh(4, 4)

# Refine mesh close to x = (0.5, 0.5)
p = Point(0.5, 0.5)
for i in range(2):

    print("marking for refinement")

    # Mark cells for refinement
    cell_markers = MeshFunction("bool", mesh, mesh.topology().dim()) # first argument is type of input, second is mesh, third is dimension.
    for c in cells(mesh):
        if c.midpoint().distance(p) < 0.2: # checking if the midpoint of the cell is close to the point p
            cell_markers[c] = True
        else:
            cell_markers[c] = False

    # Refine mesh
    mesh = refine(mesh, cell_markers)

    # Plot mesh
    plt.figure()
    plot(mesh)
    plt.show()


coordinates = mesh.coordinates()

V = FunctionSpace(mesh, 'P', 1)
u = interpolate(Expression('x[0] + x[1]', degree=1), V)

plot(u)
nodal_values = u.vector()[:]

print(nodal_values)

vertex_values = u.compute_vertex_values()

for i, x in enumerate(coordinates):
    print('vertex %d: vertex_values[%d] = %g\tu(%s) = %g' %(i, i, vertex_values[i], x, u(x)))


plt.figure()
plot(mesh)
plt.show()






