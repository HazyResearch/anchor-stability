#ifndef PYDENSEMATRIX_H
#define PYDENSEMATRIX_H

#include <stddef.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "datastructures/DenseMatrix.h"

namespace py = pybind11;

template <class T>
class PyDenseMatrix {
 public:
  PyDenseMatrix(py::array b) : b(b) {
    /* Request a buffer descriptor from Python */
    py::buffer_info info = b.request();

    /* Some sanity checks ... */
    if (info.format != py::format_descriptor<T>::format()) {
      char error_msg [100];
      sprintf(error_msg, "Incompatible format: expected an array of type %s", typeid(T).name());
      throw std::runtime_error(error_msg);
    }

    if (info.ndim != 2) {
      throw std::runtime_error("Incompatible buffer dimension!");
    }

    dense_matrix = DenseMatrix<T>(info.shape[0], info.shape[1]);
    memcpy(dense_matrix.data(), info.ptr, sizeof(T) * (size_t) (dense_matrix.n_rows() * dense_matrix.n_cols()));
  }

  PyDenseMatrix(DenseMatrix<T>& dm) : dense_matrix(std::move(dm)) {}

  T *data() { return dense_matrix.data(); }
  size_t n_rows() const { return dense_matrix.n_rows(); }
  size_t n_cols() const { return dense_matrix.n_cols(); }
  py::array_t<T> numpy() {
    py::array v = py::array(py::buffer_info(
      dense_matrix.data(),                 /* Pointer to buffer */
      sizeof(T),                           /* Size of one scalar */
      py::format_descriptor<T>::format(),  /* Python struct-style format descriptor */
      2,                                   /* Number of dimensions */
      { dense_matrix.n_rows(), dense_matrix.n_cols()}, /* Buffer dimensions */
      {sizeof(T) * size_t(dense_matrix.n_cols()), sizeof(T)} /* Strides (in bytes) for each index */
    ));
    return v;
  }
  DenseMatrix<T> dense_matrix;

 private:
  py::array b;  // adds a reference to Python object so it is not deleted.
};

#endif