#include "solver.h"
#include <cblas.h>
#include <lapacke.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <iostream>
#include <memory>
#include "common.h"
#include "utils/utils.h"
#include <set>

namespace solver {

DenseMatrix<double> cpu_sgd(
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
    const size_t input_batch_size,
    const double lr,
    const double beta,
    const double lambda,
    const double lr_decay,
    const size_t n_threads,
    const std::set<size_t> freeze_idx_set,
    const size_t freeze_n_epochs,
    DenseMatrix<double> previous_model,
    size_t& used_epochs) {
  const auto solver_time = timer::start_clock();

  ThreadPool::initialize_thread_pool(n_threads);
  srand(seed);


  double eta = lr;

  // Update batch_size to handle original batch_size (input_batch_size) > nnz.
  size_t batch_size = std::min(input_batch_size, csr.nnz());
  LOG("Batch size:\t" + std::to_string(batch_size));
  LOG("Learning rate:\t" + std::to_string(eta));
  LOG("Beta:\t" + std::to_string(beta));
  LOG("Lambda:\t" + std::to_string(lambda));
  LOG("Learning rate decay:\t" + std::to_string(lr_decay));

  // Buffer for the r from qr decomposition.
  DenseMatrix<double> r(n_dimensions, n_dimensions);

  // Initialize embeddings.
  DenseMatrix<double> return_X = utils::initialize_embeddings(
      csr.n(), n_dimensions, std::move(previous_model), r, seed);
  double *X = return_X.data();

  LOG("LOSS:\t" + std::to_string(common::loss(coo, X, n_dimensions)));
  // Buffer for the previous epoch's embedding.
  DenseMatrix<double> return_X_pre(csr.n(), n_dimensions);
  return_X_pre.zero();
  double *X_pre = return_X_pre.data();

  double *const par_dx =
      (double *)utils::get_parallel_buffer(n_threads, n_dimensions * 2 *
                                          batch_size * sizeof(double));

  double *const par_x_snapshot =
      (double *)utils::get_parallel_buffer(n_threads, n_dimensions * 2 *
                                          batch_size * sizeof(double));

  size_t epoch = 0;
  double pre_total_error = std::numeric_limits<size_t>::max();
  double total_error = 0.0;

  DenseMatrix<double> X_procrustes;

  for (; epoch < n_epochs; ++epoch) {
    const auto epoch_timer = timer::start_clock();
    if (epoch > 0)
      eta = lr / (int(epoch / lr_decay) + 1.0);
    auto total_cost_reducer = par::reducer<double>(
        0, [](const double a, const double b) { return a + b; });

    const size_t num_batches = ceil(csr.nnz() / batch_size);
    par::for_range(0, num_batches, [&](const size_t tid, const size_t batch) {
      const size_t batch_start_index = batch * batch_size;
      const size_t cur_batch_size = ((batch + 1) * batch_size <= csr.nnz())
                                        ? batch_size
                                        : csr.nnz() - batch * batch_size;

      double *const dx = &par_dx[tid * n_dimensions * 2 * batch_size];
      double *const x_snapshot =
          &par_x_snapshot[tid * n_dimensions * 2 * batch_size];

      int *const batch_rows = &coo.rowind()[batch_start_index];
      int *const batch_cols = &coo.colind()[batch_start_index];
      double *const batch_vals = &coo.val()[batch_start_index];

      // Derivative values are placed into dx.
      total_cost_reducer.update(tid,
          common::compute_derivative(cur_batch_size, n_dimensions, eta,
                                    batch_rows, batch_cols, batch_vals, X, dx));

      // Snapshot X.
      // For regularization and momentum, we need snapshot the current X.
      // We cannot use X since batch updates might change the same X_i
      // more than once and cause the wrong regularization.
      // Momemtum needs it to calcuate the difference between
      // the current X and previous X.
      // Snapshot rows in batch is cheaper than snapshot the whole matrix
      // since we only change the rows in the batch.
      if (lambda || beta) {
        for (size_t i = 0; i < cur_batch_size; ++i) {
          const size_t row = batch_rows[i];
          const size_t col = batch_cols[i];

          double *const x_row = &X[row * n_dimensions];
          double *const x_col = &X[col * n_dimensions];

          for (size_t j = 0; j < n_dimensions; ++j) {
            x_snapshot[2 * i * n_dimensions + j] = x_row[j];
            x_snapshot[(2 * i + 1) * n_dimensions + j] = x_col[j];
          }
        }
      }

      // Update step.
      for (size_t i = 0; i < cur_batch_size; ++i) {
        const size_t row = batch_rows[i];
        const size_t col = batch_cols[i];
        const double value = batch_vals[i];

        double *const x_row = &X[row * n_dimensions];
        double *const x_col = &X[col * n_dimensions];

        bool update_row = true;
        bool update_col = true;
        std::set<std::size_t>::iterator it = freeze_idx_set.find(row);
        if(it != freeze_idx_set.end()) update_row = false;
        it = freeze_idx_set.find(col);
        if(it != freeze_idx_set.end()) update_col = false;

        for (size_t j = 0; j < n_dimensions; ++j) {
          if (update_row || epoch >= freeze_n_epochs) x_row[j] += dx[2 * i * n_dimensions + j] / cur_batch_size;
          if (update_col || epoch >= freeze_n_epochs) x_col[j] += dx[(2 * i + 1) * n_dimensions + j] / cur_batch_size;
        }

        if (lambda) {
          for (size_t j = 0; j < n_dimensions; ++j) {
            if (update_row || epoch >= freeze_n_epochs) x_row[j] -= eta * lambda * x_snapshot[2 * i * n_dimensions + j];
            if (update_col || epoch >= freeze_n_epochs) x_col[j] -= eta * lambda * x_snapshot[(2 * i + 1) * n_dimensions + j];
          }
        }

        if (beta) {
          for (size_t j = 0; j < n_dimensions; ++j) {
            if (update_row || epoch >= freeze_n_epochs) x_row[j] += beta * (x_snapshot[2 * i * n_dimensions + j] - X_pre[row * n_dimensions + j]);
            if (update_col || epoch >= freeze_n_epochs) x_col[j] += beta * (x_snapshot[(2 * i + 1) * n_dimensions + j] - X_pre[col * n_dimensions + j]);
          }
        }
      }

      if (beta) {
        // Update X_pre.
        // Update previous X using the snapshot X (x_snapshot).
        for (size_t i = 0; i < cur_batch_size; ++i) {
          const size_t row = batch_rows[i];
          const size_t col = batch_cols[i];

          double *const x_row = &X_pre[row * n_dimensions];
          double *const x_col = &X_pre[col * n_dimensions];

          bool update_row = true;
          bool update_col = true;
          std::set<std::size_t>::iterator it = freeze_idx_set.find(row);
          if(it != freeze_idx_set.end()) update_row = false;
          it = freeze_idx_set.find(col);
          if(it != freeze_idx_set.end()) update_col = false;

          for (size_t j = 0; j < n_dimensions; ++j) {
           if (update_row || epoch >= freeze_n_epochs) x_row[j] = x_snapshot[2 * i * n_dimensions + j];
           if (update_col || epoch >= freeze_n_epochs) x_col[j] = x_snapshot[(2 * i + 1) * n_dimensions + j];
          }
        }
      }
    });

    total_error = total_cost_reducer.evaluate(0) / csr.nnz();

    if (save_epochs != 0 && (epoch + 1) % save_epochs == 0){
        if (epoch < freeze_n_epochs) {
          utils::log_epoch_information(
            return_X, vocab, output_file, epoch, n_epochs, save_epochs, log_epochs,
            total_error, tol, eta, epoch_timer, [&]() {
              return common::loss(coo, X, n_dimensions);
            });
        } else {
          X_procrustes = utils::orthogonal_procrustes(return_X);
          utils::log_epoch_information(
            X_procrustes, vocab, output_file, epoch, n_epochs, save_epochs, log_epochs,
            total_error, tol, eta, epoch_timer, [&]() {
              return common::loss(coo, X, n_dimensions);
            });
        }
    } else {
      utils::log_epoch_information(
          X_procrustes, vocab, output_file, epoch, n_epochs, save_epochs, log_epochs,
          total_error, tol, eta, epoch_timer, [&]() {
            return common::loss(coo, X, n_dimensions);
          });
    }
    // Convergence check!
    if(std::isnan(total_error)){
      LOG("Gradients exploded. Returning!");
      break;
    }
    if (total_error <= tol) {
      LOG("Convergence condition met. Returning!");
      break;
    }

    pre_total_error = total_error;

  }

  timer::stop_clock("SOLVER TIME", solver_time);

  used_epochs += epoch;

  // Procrustes
  const auto orthogonal_procrustes = timer::start_clock();

  X_procrustes = utils::orthogonal_procrustes(return_X);
  timer::stop_clock("PROCRUSTES TIME", orthogonal_procrustes);

  const auto save_to_file = timer::start_clock();

  if (epoch == n_epochs){
    utils::save_to_file(return_X, vocab,
                      output_file + "." +
                          std::to_string(epoch) + ".final_orig");
    utils::save_to_file(X_procrustes, vocab,
                      output_file + "." +
                          std::to_string(epoch) + ".final");
  }
  else{
    utils::save_to_file(return_X, vocab,
                      output_file + "." +
                          std::to_string(epoch+1) + ".final_orig");
    utils::save_to_file(X_procrustes, vocab,
                      output_file + "." +
                          std::to_string(epoch+1) + ".final");
  }
  timer::stop_clock("WRITING FILE", save_to_file);

  free(par_dx);
  ThreadPool::delete_thread_pool();

  return X_procrustes;
}

}  // en
