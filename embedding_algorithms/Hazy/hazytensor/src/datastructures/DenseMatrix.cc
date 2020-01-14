#include "DenseMatrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cblas.h>
#include <lapacke.h>
#include "utils/utils.h"
#include <memory>
#include <random>

template <class T>
DenseMatrix<T>::DenseMatrix(const size_t n_rows, const size_t n_cols)
    : n_rows_(n_rows), n_cols_(n_cols) {
  data_ = std::unique_ptr<T>((T *)malloc(n_rows * n_cols * sizeof(T)));
}

template <class T>
DenseMatrix<T>::DenseMatrix(const size_t n_rows, const size_t n_cols,
                            std::unique_ptr<T> data)
    : n_rows_(n_rows), n_cols_(n_cols), data_(std::move(data)) {}

template <class T>
void DenseMatrix<T>::rand_int_init(const int mod_num, const size_t seed){
  srand(seed);
  T* A = data_.get();
#pragma omp parallel for
  for (size_t i = 0; i < n_rows_ * n_cols_; ++i) A[i] = ((int)rand()) % mod_num;
}

template <class T>
void DenseMatrix<T>::rand_uniform_init(const size_t seed) {
  T* A = data_.get();
  std::default_random_engine generator;
  generator.seed(seed);
  std::normal_distribution<T> distribution(0, 1);
#pragma omp parallel for
  for (size_t i = 0; i < n_rows_ * n_cols_; ++i) A[i] = distribution(generator);
}

template <class T>
void DenseMatrix<T>::rand_uniform_init_no_par(const size_t seed) {
  T* A = data_.get();
  std::default_random_engine generator;
  generator.seed(seed);
  std::normal_distribution<T> distribution(0, 1. / n_cols_);
  for (size_t i = 0; i < n_rows_ * n_cols_; ++i) A[i] = distribution(generator);
}

template <class T>
void DenseMatrix<T>::rand_xavier_init(const size_t seed){
  T* A = data_.get();
  std::default_random_engine generator;
  generator.seed(seed);
  double bound = std::sqrt(6.0) / std::sqrt(n_rows_ + n_cols_);
  std::uniform_real_distribution<double> distribution(-bound, bound);
#pragma omp parallel for
  for (size_t i = 0; i < n_rows_ * n_cols_; ++i) A[i] = distribution(generator);
}

template <class T>
void DenseMatrix<T>::rand_xavier_init_no_par(const size_t seed){
  T* A = data_.get();
  std::default_random_engine generator;
  generator.seed(seed);
  double bound = std::sqrt(6.0) / std::sqrt(n_rows_ + n_cols_);
  std::uniform_real_distribution<double> distribution(-bound, bound);
  for (size_t i = 0; i < n_rows_ * n_cols_; ++i) A[i] = distribution(generator);
}

template <class T>
DenseMatrix<T> DenseMatrix<T>::transpose() const {
  T *data = data_.get();
  T *new_data = (T *)malloc(n_rows_ * n_cols_ * sizeof(T));

  par::for_range(0, n_cols_, [&](const size_t tid, const size_t j) {
    for (size_t i = 0; i < n_rows_; ++i) {
      new_data[j * n_rows_ + i] = data[i * n_cols_ + j];
    }
  });

  return DenseMatrix<T>(n_cols_, n_rows_,
                        std::move(std::unique_ptr<T>(new_data)));
}

template<>
void DenseMatrix<float>::qr(DenseMatrix<float> &Q_matrix,
                             DenseMatrix<float> &R_matrix) {
  ASSERT(true, "FLOAT QR IS NOT IMPLEMENTED YET");
}

template <>
void DenseMatrix<double>::qr(DenseMatrix<double> &Q_matrix,
                             DenseMatrix<double> &R_matrix) {
  // Rip out some values at the start to make life easier.
  double *const _A = data_.get();
  const size_t _m = n_rows_;
  const size_t _n = n_cols_;
  double *const _Q = Q_matrix.data();
  double *const _R = R_matrix.data();

  // Maximal rank is used by Lapacke
  const size_t rank = std::min(_m, _n);

  // Tmp Array for Lapacke
  const std::unique_ptr<double[]> tau(new double[rank]);

  // Max off diag value
  double max_off_diag_value = 0.0;

  // Calculate QR factorisations
  LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, (int)_m, (int)_n, _A, (int)_n, tau.get());
  // Copy the upper triangular Matrix R (rank x _n) into position
  for (size_t row = 0; row < rank; ++row) {
    memset(_R + row * _n, 0, row * sizeof(double));  // Set starting zeros
    memcpy(
        _R + row * _n + row, _A + row * _n + row,
        (_n - row) *
            sizeof(double));  // Copy upper triangular part from Lapack result.
  }

  // Create orthogonal matrix Q (in tmpA)
  LAPACKE_dorgqr(LAPACK_ROW_MAJOR, (int)_m, (int)rank, (int)rank, _A, (int)_n,
                 tau.get());

  // Copy Q (_m x rank) into position
  if (_m == _n) {
    memcpy(_Q, _A, sizeof(double) * (_m * _n));
  } else {
    for (size_t row = 0; row < _m; ++row) {
      memcpy(_Q + row * rank, _A + row * _n, sizeof(double) * (rank));
    }
  }
}

template<class T>
void DenseMatrix<T>::zero(){
  memset(data_.get(), 0, n_rows_ * n_cols_ * sizeof(T));
}

template<class T>
void DenseMatrix<T>::print(){
  for (size_t i = 0; i < n_rows_; ++i) {
    for (size_t j = 0; j < n_cols_; ++j) {
      std::cout << data_.get()[i * n_cols_ + j] << " ";
    }
    std::cout << std::endl;
  }
}

template<class T>
DenseVector<T> DenseMatrix<T>::get_row(const size_t row) const {
  return DenseVector<T>(n_cols_, &data_.get()[row*n_cols_]);
}


template class DenseMatrix<float>;
template class DenseMatrix<double>;
