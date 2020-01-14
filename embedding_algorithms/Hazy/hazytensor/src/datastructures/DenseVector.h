#ifndef DENSEVECTOR_H
#define DENSEVECTOR_H

#include <stddef.h>
#include <iostream>

template <class T>
class DenseVector {
 public:
  DenseVector(){}
  DenseVector(const size_t n);

  DenseVector(const size_t n, T* data);

  T* data() const { return data_; }

  size_t n() const { return n_; }

  void rand_uniform_init(const size_t seed=1234);

  T norm();
  void normalize();

  void zero();
  void print();

 private:
  size_t n_;
  T* data_;
};

#endif