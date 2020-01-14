#include "bindings/PyDenseMatrix.h"
#include "bindings/PyCSR.h"
#include "bindings/PyCOO.h"
#include "bindings/PyProcrustes.h"
#include "src/glove/vocab_count.h"
#include "src/glove/cooccur.h"
#include "src/glove/shuffle.h"
#include "src/utils/utils.h"
#include "bindings/PySolver.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(hazytensor, m) {
  m.doc() = "Python wrapper for hazy";

  m.def("vocab_count", &vocab_count::vocab_count, py::arg("corpus_file"),
        py::arg("vocab_file"), py::arg("verbose") = 0, py::arg("max_vocab") = 0,
        py::arg("min_count") = 1);

  m.def("cooccur", &cooccur::cooccur, py::arg("corpus_file"),
        py::arg("cooccur_file"), py::arg("verbose") = 2,
        py::arg("symmetric") = 1, py::arg("window_size") = 15,
        py::arg("vocab_file") = "vocab.txt", py::arg("memory") = 3.0,
        py::arg("max_product") = 0,
        py::arg("overflow_length") = 0,
        py::arg("overflow_file") = "overflow");

  m.def("shuffle", &shuffle::shuffle, py::arg("cooccur_in"),
        py::arg("cooccur_out"), py::arg("verbose") = 2, py::arg("memory") = 2.0,
        py::arg("array_size") = 0, py::arg("temp_file") = "temp_shuffle");

  m.def("solve", &PySolver::solve, py::arg("csr"), py::arg("coo"),
        py::arg("n_iterations") = 50, py::arg("n_dimensions") = 300,
        py::arg("tol") = 1e-4, py::arg("print_iterations") = 0,
        py::arg("log_iterations") = 0, py::arg("output") = "embeddings.txt",
        py::arg("vocab") = "vocab.txt", py::arg("solver") = "pi",
        py::arg("seed") = 1234, py::arg("svrg_freq") = 0,
        py::arg("batch_size") = 128, py::arg("lr") = 1e-3,
        py::arg("beta") = 0, py::arg("lambda") = 0, py::arg("lr_decay") = 0,
        py::arg("n_threads") = 4, py::arg("freeze_vocab_filename") = "",
        py::arg("freeze_n_epochs") = 0, py::arg("pre_trained") = "",
        py::arg("new_corpus") = false);

  m.def("orthogonal_procrustes", &PyProcrustes::orthogonal_procrustes,
        py::arg("matrix"), py::arg("seed") = 1234);

  m.def("procrustes_pi", &PyProcrustes::procrustes_pi, py::arg("matrix"));

  py::class_<PyCOO<double>>(m, "DoubleCOO", py::buffer_protocol())
      .def(py::init<py::array, py::array, py::array, COO<double>&>())
      .def("from_file", &PyCOO<double>::from_file)
      .def("from_csr", &PyCOO<double>::from_csr)
      .def("shuffle_inplace", &PyCOO<double>::shuffle_inplace)
      .def("sample", &PyCOO<double>::sample)
      .def("row", &PyCOO<double>::row)
      .def("col", &PyCOO<double>::col)
      .def("data", &PyCOO<double>::data)
      .def("scipy", &PyCOO<double>::scipy)
      .def("print", &PyCOO<double>::print);

  py::class_<PyCSR<float>>(m, "FloatCSR", py::buffer_protocol())
      .def(py::init<py::array, py::array, py::array, CSR<float>>())
      .def("print", &PyCSR<float>::print)
      .def_static("from_coo", &PyCSR<float>::from_coo)
      .def_static("from_csr", &PyCSR<float>::from_csr)
      .def("scipy", &PyCSR<float>::scipy)
      .def("indices", &PyCSR<float>::indices)
      .def("indptr", &PyCSR<float>::indptr)
      .def("data", &PyCSR<float>::data);

  py::class_<PyCSR<double>>(m, "DoubleCSR",
                                     py::buffer_protocol())
      .def(py::init<py::array, py::array, py::array, CSR<double>>())
      .def("print", &PyCSR<double>::print)
      .def_static("from_coo", &PyCSR<double>::from_coo)
      .def_static("from_csr", &PyCSR<double>::from_csr)
      .def("scipy", &PyCSR<double>::scipy)
      .def("indices", &PyCSR<double>::indices)
      .def("indptr", &PyCSR<double>::indptr)
      .def("data", &PyCSR<double>::data)
      .def("ppmi", &PyCSR<double>::ppmi);

  py::class_<PyDenseMatrix<float>>(m, "FloatDenseMatrix", py::buffer_protocol())
      .def(py::init<py::array>())
      .def(py::init<DenseMatrix<float>&>())
      .def("numpy", &PyDenseMatrix<float>::numpy);

  py::class_<PyDenseMatrix<double>>(m, "DoubleDenseMatrix",
                                    py::buffer_protocol())
      .def(py::init<py::array>())
      .def(py::init<DenseMatrix<double>&>())
      .def("numpy", &PyDenseMatrix<double>::numpy);
}