#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stddef.h>

#include "PyCSR.h"
#include "datastructures/COO.h"

namespace py = pybind11;

namespace {

template <class T>
class PyCOO {
 public:
  PyCOO(py::array i, py::array j, py::array v,
                 COO<T>& coo)
      : i_(i), j_(j), v_(v), coo(std::move(coo)) {}

  static PyCOO<T> from_csr(const PyCSR<T>& pycsr) {

    COO<T> coo = COO<T>::from_csr(pycsr.csr);

    py::array i = py::array(py::buffer_info(
        coo.rowind(),                           /* Pointer to buffer */
        sizeof(int),                          /* Size of one scalar */
        py::format_descriptor<int>::format(), /* Python struct-style format
                                                 descriptor */
        1,                                    /* Number of dimensions */
        {coo.nnz()},                            /* Buffer dimensions */
        {sizeof(int)} /* Strides (in bytes) for each index */
        ));

    py::array j = py::array(py::buffer_info(
        coo.colind(),                           /* Pointer to buffer */
        sizeof(int),                          /* Size of one scalar */
        py::format_descriptor<int>::format(), /* Python struct-style format
                                                 descriptor */
        1,                                    /* Number of dimensions */
        {coo.nnz()},                            /* Buffer dimensions */
        {sizeof(int)} /* Strides (in bytes) for each index */
        ));

    py::array v = py::array(py::buffer_info(
        coo.val(),                           /* Pointer to buffer */
        sizeof(double),                          /* Size of one scalar */
        py::format_descriptor<double>::format(), /* Python struct-style format
                                                 descriptor */
        1,                                    /* Number of dimensions */
        {coo.nnz()},                            /* Buffer dimensions */
        {sizeof(double)} /* Strides (in bytes) for each index */
        ));

    return PyCOO(i, j, v, coo);
  }

  static PyCOO<T> from_file(const std::string filename) {
    COO<T> coo = COO<T>::from_file(filename);

    py::array i = py::array(py::buffer_info(
        coo.rowind(),                           /* Pointer to buffer */
        sizeof(int),                          /* Size of one scalar */
        py::format_descriptor<int>::format(), /* Python struct-style format
                                                 descriptor */
        1,                                    /* Number of dimensions */
        {coo.nnz()},                            /* Buffer dimensions */
        {sizeof(int)} /* Strides (in bytes) for each index */
        ));

    py::array j = py::array(py::buffer_info(
        coo.colind(),                           /* Pointer to buffer */
        sizeof(int),                          /* Size of one scalar */
        py::format_descriptor<int>::format(), /* Python struct-style format
                                                 descriptor */
        1,                                    /* Number of dimensions */
        {coo.nnz()},                            /* Buffer dimensions */
        {sizeof(int)} /* Strides (in bytes) for each index */
        ));

    py::array v = py::array(py::buffer_info(
        coo.val(),                           /* Pointer to buffer */
        sizeof(double),                          /* Size of one scalar */
        py::format_descriptor<double>::format(), /* Python struct-style format
                                                 descriptor */
        1,                                    /* Number of dimensions */
        {coo.nnz()},                            /* Buffer dimensions */
        {sizeof(double)} /* Strides (in bytes) for each index */
        ));

    return PyCOO(i, j, v, coo);
  }

  PyCOO<T> sample(const size_t sample_percent, const size_t seed,
    const bool symmetric) {
    COO<T> coo_new = coo.sample(sample_percent, seed, symmetric);
    py::array i = py::array(py::buffer_info(
        coo_new.rowind(),                           /* Pointer to buffer */
        sizeof(int),                          /* Size of one scalar */
        py::format_descriptor<int>::format(), /* Python struct-style format
                                                 descriptor */
        1,                                    /* Number of dimensions */
        {coo_new.nnz()},                            /* Buffer dimensions */
        {sizeof(int)} /* Strides (in bytes) for each index */
        ));

    py::array j = py::array(py::buffer_info(
        coo_new.colind(),                           /* Pointer to buffer */
        sizeof(int),                          /* Size of one scalar */
        py::format_descriptor<int>::format(), /* Python struct-style format
                                                 descriptor */
        1,                                    /* Number of dimensions */
        {coo_new.nnz()},                            /* Buffer dimensions */
        {sizeof(int)} /* Strides (in bytes) for each index */
        ));

    py::array v = py::array(py::buffer_info(
        coo_new.val(),                           /* Pointer to buffer */
        sizeof(double),                          /* Size of one scalar */
        py::format_descriptor<double>::format(), /* Python struct-style format
                                                 descriptor */
        1,                                    /* Number of dimensions */
        {coo_new.nnz()},                            /* Buffer dimensions */
        {sizeof(double)} /* Strides (in bytes) for each index */
        ));

    return PyCOO(i, j, v, coo_new);
  }

  void shuffle_inplace(const size_t seed){coo.shuffle_inplace(seed);}

  void print(){coo.print();}

  py::array row(){return i_;}
  py::array col(){return j_;}
  py::array data(){return v_;}

  py::object scipy(){
    py::object scipy = py::module::import("scipy.sparse");
    py::object coo = scipy.attr("coo_matrix");
    return coo(py::make_tuple(data(), py::make_tuple(row(), col())));
  }
  COO<T> coo;

 private:
  py::array i_;
  py::array j_;
  py::array v_;
};

}