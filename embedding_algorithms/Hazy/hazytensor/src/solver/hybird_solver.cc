#include <cblas.h>
#include <lapacke.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <iostream>
#include <memory>
#include "solver.h"
#include "utils/utils.h"

namespace solver {
DenseMatrix<double> cpu_hybird_pi_sgd(
    const CSR<double> &cooccurrence, const COO<double> &coo,
    const size_t n_epochs, const size_t n_dimensions, const double tol,
    const size_t save_epochs, const size_t log_epochs,
    const std::string output_file, const std::vector<std::string> &vocab,
    const size_t seed, const size_t batch_size, const double lr,
    const double beta, const double lambda, const double lr_decay,
    const size_t n_threads, const std::set<size_t> freeze_idx_set,
    const size_t freeze_n_epochs, DenseMatrix<double> previous_model) {
  size_t used_epochs = 0;
  LOG("Running power method first.");
  DenseMatrix<double> pi_embedding = solver::cpu_simultaneous_power_iteration(
      cooccurrence, coo, n_epochs, n_dimensions, tol, save_epochs,
      log_epochs, output_file, vocab, seed, n_threads,
      std::move(previous_model), used_epochs, false /* debug */);
  LOG("Switching to sgd method.");
  DenseMatrix<double> sgd_embedding = solver::cpu_sgd(
      cooccurrence, coo, (n_epochs - used_epochs), n_dimensions, tol,
      save_epochs, log_epochs, output_file, vocab, seed, batch_size, lr,
      beta, lambda, lr_decay, n_threads, freeze_idx_set, freeze_n_epochs,
      std::move(pi_embedding), used_epochs);
  return pi_embedding;
}
}