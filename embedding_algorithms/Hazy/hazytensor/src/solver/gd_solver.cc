#include <cblas.h>
#include <lapacke.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include "common.h"
#include "solver.h"
#include "utils/utils.h"

#ifndef __APPLE__
#include <omp.h>
#endif


namespace solver {

DenseMatrix<double> cpu_gd(
    const CSR<double> &csr,
    const COO<double> &coo,
    const size_t n_epochs,
    const size_t n_dimensions,
    const double tol,
    const size_t save_epochs,
    const size_t log_epochs,
    const std::string output_file,
    const std::vector<std::string> &vocab,
    const size_t seed,
    const double lr,
    const size_t n_threads,
    DenseMatrix<double> previous_model,
    size_t& used_epochs) {
  const auto solver_time = timer::start_clock();

  ThreadPool::initialize_thread_pool(n_threads);
  srand(seed);

  const double eta = lr;
#ifndef __APPLE__
  omp_set_num_threads(n_threads);  // set num threads
#endif

  // Buffer for the r from qr decomposition.
  DenseMatrix<double> r(n_dimensions, n_dimensions);

  // Initialize embeddings
  DenseMatrix<double> return_X = utils::initialize_embeddings(
      csr.n(), n_dimensions, std::move(previous_model), r,
      seed);
  double *X = return_X.data();

  // Full gradient on snapshot embeddings
  double *const dx_full =
      (double *)malloc(csr.n() * n_dimensions * sizeof(double));

  size_t epoch = 0;
  double total_error = 0.0;

  for (; epoch < n_epochs; ++epoch) {
    const auto epoch_timer = timer::start_clock();

    // Calcualte full gradient
    const auto full_grad_timer = timer::start_clock();

    // Reset full gradient
    memset(dx_full, 0, csr.n() * n_dimensions * sizeof(double));

    // Calculate full gradient
    // Derivative values are placed into dx.
    double total_cost =
        common::par_compute_derivative(coo.nnz(), n_dimensions, coo.n(), eta,
                                       coo.rowind(), coo.colind(), coo.val(),
                                       X, dx_full);
    if (utils::check_print_log_condition(epoch, log_epochs,
                                         n_epochs)) {
      timer::stop_clock("FULL GRADIENT", full_grad_timer);
    }

    for (size_t j = 0; j < csr.n() * n_dimensions; ++j) {
      X[j] += dx_full[j] / coo.nnz();
    }

    total_error = total_cost / csr.nnz();
    utils::log_epoch_information(
        return_X, vocab, output_file, epoch, n_epochs, save_epochs, log_epochs,
        total_error, tol, eta, epoch_timer, [&]() {
          return common::loss(coo, X, n_dimensions);
        });

    // Convergence check!
    if (total_error <= tol) {
      LOG("Convergence condition met. Returning!");
      break;
    }
  }

  timer::stop_clock("SOLVER TIME", solver_time);

  used_epochs += epoch;

  const auto save_to_file = timer::start_clock();
  utils::save_to_file(return_X, vocab,
                      output_file + ".e" + std::to_string(total_error) + "_i" +
                          std::to_string(epoch) + ".final");
  timer::stop_clock("WRITING FILE", save_to_file);

  free(dx_full);
  ThreadPool::delete_thread_pool();

  return return_X;
}

}  // en
