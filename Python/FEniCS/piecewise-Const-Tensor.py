from dolfin import *
import matplotlib.pyplot as plt

# Read mesh from file and create function space
mesh = Mesh("mesh.xml.gz")
V = FunctionSpace(mesh, "Lagrange", 1)

# Define Dirichlet boundary (x = 0 or x = 1)
def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)

# Define conductivity expression and matrix
c00 = MeshFunction("double", mesh, "c00.xml.gz")
c01 = MeshFunction("double", mesh, "c01.xml.gz")
c11 = MeshFunction("double", mesh, "c11.xml.gz")

# Code for C++ evaluation of conductivity
conductivity_code = """

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11;

#include <dolfin/function/Expression.h>
#include <dolfin/mesh/MeshFunction.h>

class Conductivity : public dolfin::Expression
{
public:

  // Create expression with 3 components
  Conductivity() : dolfin::Expression(3) {}

  // Function for evaluating expression on each cell
  void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& cell) const override
  {
    const uint cell_index = cell.index;
    values[0] = (*c00)[cell_index];
    values[1] = (*c01)[cell_index];
    values[2] = (*c11)[cell_index];
  }

  // The data stored in mesh functions
  std::shared_ptr<dolfin::MeshFunction<double>> c00;
  std::shared_ptr<dolfin::MeshFunction<double>> c01;
  std::shared_ptr<dolfin::MeshFunction<double>> c11;

};

PYBIND11_MODULE(SIGNATURE, m)
{
  py::class_<Conductivity, std::shared_ptr<Conductivity>, dolfin::Expression>(m, "Conductivity")
    .def(py::init<>())
    .def_readwrite("c00", &Conductivity::c00)
    .def_readwrite("c01", &Conductivity::c01)
    .def_readwrite("c11", &Conductivity::c11);
}

"""

c = CompiledExpression(compile_cpp_code(conductivity_code).Conductivity(),
                       c00=c00, c01=c01, c11=c11, degree=0)
#c=Expression("const uint cell_index = cell.index;values[0] = (*c00)[cell_index];values[1] = (*c01)[cell_index];values[2] = (*c11)[cell_index];");

C = as_matrix(((c[0], c[1]), (c[1], c[2])))

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2)
a = inner(C*grad(u), grad(v))*dx
L = f*v*dx

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Plot solution
plot(u)
plt.title(r"$u(x)$",fontsize=26)
plt.show()
