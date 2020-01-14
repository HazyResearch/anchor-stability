import os

sparse_matrix_types = ["float", "double"]

dense_matrix_types = ["float", "double"]


def gen_sparse_matrix(dtype):
    return """  
py::class_<PySparseMatrix<{dtype}>>(m, "{name}SparseMatrix", py::buffer_protocol())
  .def(py::init<py::array, py::array, py::array, SparseMatrix<{dtype}>>())
  .def("print", &PySparseMatrix<{dtype}>::print)
  .def_static("from_coo", &PySparseMatrix<{dtype}>::from_coo)
  .def_static("from_csr", &PySparseMatrix<{dtype}>::from_csr)
  .def("indices", &PySparseMatrix<{dtype}>::indices)
  .def("indptr", &PySparseMatrix<{dtype}>::indptr)
  .def("data", &PySparseMatrix<{dtype}>::data);
""".replace(
        "{dtype}", dtype
    ).replace(
        "{name}", dtype.title()
    )


def gen_dense_matrix(dtype):
    return """
py::class_<PyDenseMatrix<{dtype}>>(m, "{name}DenseMatrix", py::buffer_protocol())
    .def(py::init<py::array>())
    .def("numpy", &PyDenseMatrix<{dtype}>::numpy);
""".replace(
        "{dtype}", dtype
    ).replace(
        "{name}", dtype.title()
    )


def bindings_stub(code):
    return """\
#include <pybind11/pybind11.h>
#include "bindings/PyDenseMatrix.h"
#include "bindings/PySparseMatrix.h"

namespace py = pybind11;

PYBIND11_MODULE(hazytensor, m) {
  m.doc() = "Python wrapper for hazy";

  {code}

}""".replace(
        "{code}", code
    )


def main():
    code = ""
    for sp in sparse_matrix_types:
        code += gen_sparse_matrix(sp)
    for d in dense_matrix_types:
        code += gen_dense_matrix(d)

    code = bindings_stub(code)

    file = open("bindings.cc", "w")
    file.write(code)
    file.close()
    os.system("clang-format -i bindings.cc")


if __name__ == "__main__":
    main()
