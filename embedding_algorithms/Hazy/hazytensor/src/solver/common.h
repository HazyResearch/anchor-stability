//  Commomn utility functions shared for GD, SGD, and SVRG.

#ifndef COMMON_H
#define COMMON_H

#include <memory>
#include <vector>
#include "datastructures/COO.h"
#include "datastructures/CSR.h"

namespace common {

double loss(const COO<double> &coo, double *const matrix, const size_t n);

double compute_derivative(const size_t cur_batch_size,
                          const size_t n_dimensions, const double eta,
                          const int *const batch_rows,
                          const int *const batch_cols,
                          const double *const batch_vals, const double *const X,
                          double *const dx);

double par_compute_derivative(const size_t cur_batch_size,
                              const size_t n_dimensions, const size_t n,
                              const double eta, const int *const batch_rows,
                              const int *const batch_cols,
                              const double *const batch_vals,
                              const double *const X, double *const dx);


}  // end namespace common

#endif
