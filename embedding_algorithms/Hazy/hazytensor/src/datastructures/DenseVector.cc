#include "DenseVector.h"
#include <cblas.h>
#include <lapacke.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <random>

template <class T>
DenseVector<T>::DenseVector(const size_t n) : n_(n) {
  data_ = (T *)malloc(n * sizeof(T));
}

template <class T>
DenseVector<T>::DenseVector(const size_t n, T* data)
    : n_(n), data_(data) {}

template <class T>
void DenseVector<T>::zero() {
  memset(data_, 0, n_ * sizeof(T));
}

template <class T>
void DenseVector<T>::rand_uniform_init(const size_t seed) {
  std::default_random_engine generator;
  generator.seed(seed);
  std::normal_distribution<T> distribution(0, 1);
  for (size_t i = 0; i < n_; ++i) data_[i] = distribution(generator);
}

template <>
float DenseVector<float>::norm() {
  return cblas_snrm2(n_, data_, 1);
}

template <>
double DenseVector<double>::norm() {
  return cblas_dnrm2(n_, data_, 1);
}

template <>
void DenseVector<float>::normalize() {
  float norm = cblas_snrm2(n_, data_, 1);
  for (size_t i = 0; i < n_; ++i) data_[i] /= norm;
}

template <>
void DenseVector<double>::normalize() {
  double norm = cblas_dnrm2(n_, data_, 1);
  for (size_t i = 0; i < n_; ++i) data_[i] /= norm;
}

template<class T>
void DenseVector<T>::print(){
  for (size_t i = 0; i < n_; ++i)
    std::cout << data_[i] << " ";
  std::cout << std::endl;
}

template class DenseVector<float>;
template class DenseVector<double>;