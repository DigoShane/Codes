#This code is to test if FEniCSx is working or not.
#It was installed using Conda environment using the following 3 commands.
#conda create -n fenicsx-env
#conda activate fenicsx-env
#conda install -c conda-forge fenics-dolfinx mpich pyvista


# #--------------------------------------------------------------------------------------------------
# #                  #!!xDx!! This is from an old Dolfinx poisson tutorial
# #--------------------------------------------------------------------------------------------------
# from mpi4py import MPI
# from petsc4py.PETSc import ScalarType  # type: ignore
# import numpy as np
# 
# import ufl
# from dolfinx import fem, io, mesh, plot
# from dolfinx.fem.petsc import LinearProblem
# from ufl import ds, dx, grad, inner
# 
# msh = mesh.create_rectangle(comm=MPI.COMM_WORLD,
#                             points=((0.0, 0.0), (2.0, 1.0)), n=(32, 16),
#                             cell_type=mesh.CellType.triangle)
# V = fem.functionspace(msh, ("Lagrange", 1))
# 
# facets = mesh.locate_entities_boundary(msh, dim=(msh.topology.dim - 1),
#                                        marker=lambda x: np.logical_or(np.isclose(x[0], 0.0),
#                                                                       np.isclose(x[0], 2.0)))
# 
# dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)
# 
# bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)
# 
# u = ufl.TrialFunction(V)
# v = ufl.TestFunction(V)
# x = ufl.SpatialCoordinate(msh)
# f = 10 * ufl.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02)
# g = ufl.sin(5 * x[0])
# a = inner(grad(u), grad(v)) * dx
# L = inner(f, v) * dx + inner(g, v) * ds
# 
# problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
# uh = problem.solve()
# 
# with io.XDMFFile(msh.comm, "out_poisson/poisson.xdmf", "w") as file:
#     file.write_mesh(msh)
#     file.write_function(uh)
# 
# try:
#     import pyvista
#     cells, types, x = plot.vtk_mesh(V)
#     grid = pyvista.UnstructuredGrid(cells, types, x)
#     grid.point_data["u"] = uh.x.array.real
#     grid.set_active_scalars("u")
#     plotter = pyvista.Plotter()
#     plotter.add_mesh(grid, show_edges=True)
#     warped = grid.warp_by_scalar()
#     plotter.add_mesh(warped)
#     if pyvista.OFF_SCREEN:
#         pyvista.start_xvfb(wait=0.1)
#         plotter.screenshot("uh_poisson.png")
#     else:
#         plotter.show()
# except ModuleNotFoundError:
#     print("'pyvista' is required to visualise the solution")
#     print("Install 'pyvista' with pip: 'python3 -m pip install pyvista'")
# 
# #--------------------------------------------------------------------------------------------------
# #                  #!!xDx!! This is from an old Dolfinx poisson tutorial
# #--------------------------------------------------------------------------------------------------




#--------------------------------------------------------------------------------------------------
#                  #!!xDx!! This is from "https://jsdokken.com/dolfinx-tutorial/chapter1/fundamentals_code.html"
#--------------------------------------------------------------------------------------------------

from mpi4py import MPI
from dolfinx import mesh
domain = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8, mesh.CellType.quadrilateral)

from dolfinx.fem import FunctionSpace
V = FunctionSpace(domain, ("Lagrange", 1))

from dolfinx import fem
uD = fem.Function(V)
uD.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)

import numpy
# Create facet to cell connectivity required to determine boundary facets
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)

boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(uD, boundary_dofs)

import ufl
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

from dolfinx import default_scalar_type
f = fem.Constant(domain, default_scalar_type(-6))

a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx

from dolfinx.fem.petsc import LinearProblem
problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

V2 = fem.FunctionSpace(domain, ("Lagrange", 2))
uex = fem.Function(V2)
uex.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)

L2_error = fem.form(ufl.inner(uh - uex, uh - uex) * ufl.dx)
error_local = fem.assemble_scalar(L2_error)
error_L2 = numpy.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))

error_max = numpy.max(numpy.abs(uD.x.array-uh.x.array))
# Only print the error on one process
if domain.comm.rank == 0:
    print(f"Error_L2 : {error_L2:.2e}")
    print(f"Error_max : {error_max:.2e}")

import pyvista
print(pyvista.global_theme.jupyter_backend)

from dolfinx import plot
pyvista.start_xvfb()
topology, cell_types, geometry = plot.vtk_mesh(domain, tdim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("fundamentals_mesh.png")

u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)

u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
u_grid.point_data["u"] = uh.x.array.real
u_grid.set_active_scalars("u")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, show_edges=True)
u_plotter.view_xy()
if not pyvista.OFF_SCREEN:
    u_plotter.show()

warped = u_grid.warp_by_scalar()
plotter2 = pyvista.Plotter()
plotter2.add_mesh(warped, show_edges=True, show_scalar_bar=True)
if not pyvista.OFF_SCREEN:
    plotter2.show()

from dolfinx import io
from pathlib import Path
results_folder = Path("results")
results_folder.mkdir(exist_ok=True, parents=True)
filename = results_folder / "fundamentals"
with io.VTXWriter(domain.comm, filename.with_suffix(".bp"), [uh]) as vtx:
    vtx.write(0.0)
with io.XDMFFile(domain.comm, filename.with_suffix(".xdmf"), "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)

#--------------------------------------------------------------------------------------------------
#                  #!!xDx!! This is from "https://jsdokken.com/dolfinx-tutorial/chapter1/fundamentals_code.html"
#--------------------------------------------------------------------------------------------------
