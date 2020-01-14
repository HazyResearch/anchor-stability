#ifndef PYCSR_H
#define PYCSR_H

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stddef.h>

#include "datastructures/CSR.h"
#include "datastructures/COO.h"

namespace py = pybind11;

namespace {

template <class T>
void set_pointers_and_check_types(size_t& nnz, int** indptr, int** indices,
                                  T** data, py::array i, py::array j,
                                  py::array v) {
  py::buffer_info i_info = i.request();
  py::buffer_info j_info = j.request();
  py::buffer_info v_info = v.request();

  nnz = v_info.shape[0];

  /* Some sanity checks ... */
  if (i_info.format != py::format_descriptor<int>::format())
    throw std::runtime_error("Incompatible format: expected an int array!");

  if (j_info.format != py::format_descriptor<int>::format())
    throw std::runtime_error("Incompatible format: expected an int array!");

  if (v_info.format != py::format_descriptor<T>::format())
    throw std::runtime_error("Incompatible format: expected an " +
                             py::format_descriptor<T>::format() + " array!");

  *indptr = (int*)i_info.ptr;
  *indices = (int*)j_info.ptr;
  *data = (T*)v_info.ptr;
}
}

template <class T>
class PyCSR {
 public:
  PyCSR(py::array i, py::array j, py::array v,
                 CSR<T> csr)
      : i(i), j(j), v(v), csr(csr) {}

  static PyCSR<T> from_csr(int n, int nnz, py::array i, py::array j,
                                    py::array v) {
    int *indptr, *indices;
    T* data;
    size_t _nnz;
    set_pointers_and_check_types<T>(_nnz, &indptr, &indices, &data, i, j, v);

    //TODO add a graceful transfer for different scipy types. If the matrix
    //is large enough this will be an int64. As is this code will only 
    //work with sparse matrices that are stored with int pointers not int64.
    size_t* actual_ind_pointer = (size_t*)malloc(sizeof(size_t)*(n+1));
    #pragma omp for //simd
    for(size_t i = 0; i < n+1; ++i){
      actual_ind_pointer[i] = indices[i];
    }

    assert(_nnz == nnz);

    //TODO: Make i, j, and v python arrays that point to the CSR.
    auto sm =
        CSR<T>::from_csr(n, nnz, actual_ind_pointer, indices, data);
    return PyCSR<T>(i, j, v, sm);
  }

  static PyCSR<T> from_coo(py::array i, py::array j,
                                    py::array v) {
    int *_i, *_j;
    T* data;
    size_t nnz;
    set_pointers_and_check_types<T>(nnz, &_i, &_j, &data, i, j, v);

    size_t n = *std::max_element(_i, _i+nnz) + 1;

    auto sm = CSR<T>::from_coo(n, nnz, _i, _j, data);

    auto ii = py::array(py::buffer_info(sm.indptr(), sizeof(size_t),
                                       py::format_descriptor<size_t>::format(), 1,
                                       {n+1}, {sizeof(size_t)}));

    auto jj = py::array(py::buffer_info(sm.indices(), sizeof(int),
                                       py::format_descriptor<int>::format(), 1,
                                       {nnz}, {sizeof(int)}));

    auto vv = py::array(py::buffer_info(sm.data(), sizeof(T),
                                       py::format_descriptor<T>::format(), 1,
                                       {nnz}, {sizeof(T)}));

    return PyCSR<T>(ii, jj, vv, sm);
  }

  py::object scipy(){
    py::object scipy = py::module::import("scipy.sparse");
    py::object csr = scipy.attr("csr_matrix");
    return csr(py::make_tuple(data(), indices(), indptr()));
  }

  void print(){csr.print();}

  py::array indptr(){return i;}
  py::array indices(){return j;}
  py::array data(){return v;}

  void ppmi(){return csr.ppmi();}

  CSR<T> csr;
 private:
  py::array i;
  py::array j;
  py::array v;
};

#endif