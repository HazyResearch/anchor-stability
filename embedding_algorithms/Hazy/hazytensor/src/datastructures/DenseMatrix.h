#ifndef DENSEMATRIX_H
#define DENSEMATRIX_H

#include <memory>
#include <stddef.h>
#include <iostream>
#include "datastructures/DenseVector.h"

template <class T>
class DenseMatrix {
 public:
  DenseMatrix(){}
  DenseMatrix(const size_t rows, const size_t cols);

  DenseMatrix(const size_t rows, const size_t cols, std::unique_ptr<T> data);

  T* data() const { return data_.get(); }

  size_t n_rows() const { return n_rows_; }
  size_t n_cols() const { return n_cols_; }

  DenseMatrix<T> transpose() const;

  void zero();

  void print();

  DenseVector<T> get_row(const size_t i) const;

  //  Init matrix to random integer values in the range (0,mod_number-1).
  //  Initialization is done in parallel and won't produce deterministic
  //  results.
  void rand_int_init(const int mod_number, const size_t seed=1234);

  //  Init matrix to random T values from the Gaussian distribution (0,1).
  //  Initialization is done in parallel.
  void rand_uniform_init(const size_t seed=1234);

  // Init matrix to random T values from the Gaussian distribution (0,1).
  //  Initialization is done in parallel.
  void rand_uniform_init_no_par(const size_t seed=1234);

  //  Init matrix using Xavier uniform initializer where it draws samples
  //  from a uniform distribution within [-limit, limit] where limit
  //  is sqrt(6 / (n_rows_ + n_cols_)).
  //  Initialization is done in serial.
  void rand_xavier_init_no_par(const size_t seed=1234);

  //  Init matrix using Xavier uniform initializer where it draws samples
  //  from a uniform distribution within [-limit, limit] where limit
  //  is sqrt(6 / (n_rows_ + n_cols_)).
  //  Initialization is done in parallel.
  void rand_xavier_init(const size_t seed=1234);

  //  Calls lapacke qr decompostion methods: (q, r) = qr(A)
  //  The values in Q_matrix and R_matrix are overwritten.
  void qr(DenseMatrix<T>& Q, DenseMatrix<T>& R);

  //  Returns whether this matrix ever had data allocated or not.
  bool is_valid() { return data_ ? true : false; }

protected:
  std::unique_ptr<T> take_ownership_of_data() { return std::move(data_); }

 private:
  size_t n_rows_, n_cols_;
  std::unique_ptr<T> data_;
};

#endif
