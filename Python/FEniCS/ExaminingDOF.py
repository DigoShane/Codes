from fenics import *
mesh = UnitSquareMesh(2, 2)
coordinates = mesh.coordinates()

V = FunctionSpace(mesh, 'P', 1)
u = interpolate(Expression('x[0] + x[1]', degree=1), V)

plot(u)
nodal_values = u.vector()[:]

print(nodal_values)

vertex_values = u.compute_vertex_values()

for i, x in enumerate(coordinates):
    print('vertex %d: vertex_values[%d] = %g\tu(%s) = %g' %(i, i, vertex_values[i], x, u(x)))

v2d = vertex_to_dof_map(V)

element = V.element()
dofmap = V.dofmap()
for cell in cells(mesh):
    print(element.tabulate_dof_coordinates(cell))
    print(dofmap.cell_dofs(cell.index()))

def normalize_solution(u):
    "Normalize u: return u divided by max(u)"
    u_array = u.vector().array()
    u_max = np.max(np.abs(u_array))
    u_array /= u_max
    u.vector()[:] = u_array
    #u.vector().set_local(u_array)  # alternative
    return u





