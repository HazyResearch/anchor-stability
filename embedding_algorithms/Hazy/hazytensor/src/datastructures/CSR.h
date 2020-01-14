#ifndef CSR_H
#define CSR_H

#include <stddef.h>
#include <iostream>
#include <math.h>

template <class T>
class CSR {
 public:
  CSR(size_t _n, size_t _nnz, size_t *_indptr, int *_indices, T *_data)
      : n_(_n), nnz_(_nnz), indptr_(_indptr), indices_(_indices), data_(_data) {}

  ~CSR(){}

  void print() const;

  void ppmi();
  size_t n() const {return n_;}
  size_t nnz() const {return nnz_;}
  size_t* indptr() const {return indptr_;}
  int* indices() const {return indices_;}
  T* data() const {return data_;}
  bool find(const size_t i, const size_t j) const;

  static  CSR<T> from_csr(size_t _n, size_t _nnz, size_t *_indptr,
                                  int *_indices, T *_data) {
    return CSR<T>(_n, _nnz, _indptr, _indices, _data);
  }

  static CSR<T> from_coo(size_t _n, size_t _nnz, int *_i, int *_j,
                                  T *_data, bool sort_rows = true);


  template<typename F>
  inline void foreach(F f) const {
    for (size_t i = 0; i < n_; ++i) {
      for (int rs = indptr_[i]; rs < indptr_[i + 1]; ++rs) {
        f(i, indices_[rs], data_[rs]);
      }
    }
  }

  template<typename F>
  inline void parforeach(F f) const {
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < n_; ++i) {
      for (int rs = indptr_[i]; rs < indptr_[i + 1]; ++rs) {
        f(i, indices_[rs], data_[rs]);
      }
    }
  }


 private:
  size_t n_;
  size_t nnz_;

  size_t* indptr_;
  int* indices_;
  T* data_;
};

#endif