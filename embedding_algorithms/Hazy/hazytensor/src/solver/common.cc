//  Commomn utility functions shared for GD, SGD, and SVRG.

#include "common.h"
#include <cblas.h>
#include <lapacke.h>
#include <math.h>
#include <stdio.h>
#include <mutex>
#include <vector>
#include "utils/utils.h"

#ifndef __APPLE__
#include <omp.h>
#endif

namespace common {

double loss(const COO<double> &coo, double *const X,
                    const size_t n_dimensions) {
  auto total_cost_reducer = par::reducer<double>(
      0, [](const double a, const double b) { return a + b; });
  par::for_range(0, coo.nnz(), [&](const size_t tid, const size_t i) {
    const int row = coo.rowind()[i];
    const int col = coo.colind()[i];
    const double value = coo.val()[i];

    const double *const x_row = &X[row * n_dimensions];
    const double *const x_col = &X[col * n_dimensions];

    const double pred = cblas_ddot(n_dimensions, x_row, 1, x_col, 1);

    const double error = pred - value;
    total_cost_reducer.update(tid, (error * error));
  });
  return total_cost_reducer.evaluate(0) / coo.nnz();
}

double compute_derivative(const size_t cur_batch_size,
                          const size_t n_dimensions, const double eta,
                          const int *const batch_rows,
                          const int *const batch_cols,
                          const double *const batch_vals, const double *const X,
                          double *const dx) {
  double total_cost = 0.0;
  for (size_t i = 0; i < cur_batch_size; ++i) {
    const int row = batch_rows[i];
    const int col = batch_cols[i];
    const double value = batch_vals[i];

    const double *const x_row = &X[row * n_dimensions];
    const double *const x_col = &X[col * n_dimensions];

    const double pred = cblas_ddot(n_dimensions, x_row, 1, x_col, 1);

    const double error = pred - value;
    const double step = -eta * error;

    for (size_t j = 0; j < n_dimensions; ++j) {
      dx[2 * i * n_dimensions + j] = step * x_col[j];
      dx[(2 * i + 1) * n_dimensions + j] = step * x_row[j];
    }

    total_cost += error * error;

    ASSERT(!std::isnan(error * error) && !std::isnan(total_cost),
           std::to_string(error) + "  " + std::to_string(total_cost));
  }
  return total_cost;
}

double par_compute_derivative(const size_t cur_batch_size,
                              const size_t n_dimensions, const size_t n,
                              const double eta, const int *const batch_rows,
                              const int *const batch_cols,
                              const double *const batch_vals,
                              const double *const X, double *const dx) {
  
  std::vector<std::mutex> mutexes(cur_batch_size);
  auto total_cost_reducer = par::reducer<double>(
      0, [](const double a, const double b) { return a + b; });
  par::for_range(0, cur_batch_size, [&](const size_t tid, const size_t i){

    const int row = batch_rows[i];
    const int col = batch_cols[i];
    const double value = batch_vals[i];

    const double *const x_row = &X[row * n_dimensions];
    const double *const x_col = &X[col * n_dimensions];

    const double pred = cblas_ddot(n_dimensions, x_row, 1, x_col, 1);

    const double error = pred - value;
    const double step = -eta * error;

    mutexes.at(row).lock();
    for (size_t j = 0; j < n_dimensions; ++j) {
      dx[row * n_dimensions + j] += step * x_col[j];
    }
    mutexes.at(row).unlock();

    mutexes.at(col).lock();
    for (size_t j = 0; j < n_dimensions; ++j) {
      dx[col * n_dimensions + j] += step * x_row[j];
    }
    mutexes.at(col).unlock();

    total_cost_reducer.update(tid, (error * error));
  });
  return total_cost_reducer.evaluate(0);
}

}
