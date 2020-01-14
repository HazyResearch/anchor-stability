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

DenseMatrix<double> cpu_svrg(
    const CSR<double> &csr,
    const COO<double> &coo,
    const size_t n_epochs,
    const size_t n_dimensions,
    const double tol,
    const size_t T,
    const size_t save_epochs,
    const size_t log_epochs,
    const std::string output_file,
    const std::vector<std::string> &vocab,
    const size_t seed,
    const size_t input_batch_size,
    const double lr,
    const size_t n_threads,
    DenseMatrix<double> previous_model,
    size_t& used_epochs) {
  const auto solver_time = timer::start_clock();

  ThreadPool::initialize_thread_pool(n_threads);
  srand(seed);

  const double eta = lr;
  // Update batch_size to handle original batch_size (input_batch_size) > nnz.
  size_t batch_size = std::min(input_batch_size, csr.nnz());
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

  // Snapshot embeddings
  double *const X_tilde =
      (double *)malloc(csr.n() * n_dimensions * sizeof(double));

  // Batch gradient on current embeddings
  double *const par_dx =
      (double *)utils::get_parallel_buffer(n_threads, n_dimensions * 2 *
                                          batch_size * sizeof(double));

  // Batch gradient on snapshot embeddings
  double *const par_dx_tilde =
      (double *)utils::get_parallel_buffer(n_threads, n_dimensions * 2 *
                                          batch_size * sizeof(double));

  // Full gradient on snapshot embeddings
  double *const dx_full =
      (double *)malloc(csr.n() * n_dimensions * sizeof(double));

  size_t epoch = 0;
  double total_error = 0.0;
  double loss = 0.0;

  size_t* counters = (size_t*)malloc(coo.n()*sizeof(size_t));

  DenseMatrix<double> X_procrustes;

  for (; epoch < n_epochs; ++epoch) {
    const auto epoch_timer = timer::start_clock();
    const auto itr_timer = timer::start_clock();

    double total_cost = 0.0;

    // Snapshot the current embedding and calcualte full gradient
    if (epoch == 0 || ((epoch + 1) % T) == 0) {
      const auto full_grad_timer = timer::start_clock();

      if (utils::check_print_log_condition(epoch, log_epochs,
                                           n_epochs)) {
        std::cout << "Snaphot embeddings at epoch " << epoch + 1
                  << std::endl;
      }

      // Snapshot the embedding
      memcpy(X_tilde, X, csr.n() * n_dimensions * sizeof(double));
      // Reset full gradient
      memset(dx_full, 0, csr.n() * n_dimensions * sizeof(double));

      // Calculate full gradient on snapshot
      // Derivative values are placed into dx.
      common::par_compute_derivative(coo.nnz(), n_dimensions, coo.n(), eta,
                             coo.rowind(), coo.colind(), coo.val(),
                             X_tilde, dx_full);
      if (utils::check_print_log_condition(epoch, log_epochs,
                                           n_epochs)) {
        timer::stop_clock("FULL GRADIENT", full_grad_timer);
      }
    }

    memset(counters, 0, csr.n() * sizeof(size_t));
    std::atomic<size_t> current_epoch;
    current_epoch = 0;
    std::vector<std::mutex> mutexes(coo.n());

    // Calculate the batch gradient and update embeddings
    auto total_cost_reducer = par::reducer<double>(
        0, [](const double a, const double b) { return a + b; });
    const size_t num_batches = ceil(csr.nnz() / batch_size);
    par::for_range(0, num_batches, [&](const size_t tid, const size_t batch) {
      const size_t batch_start_index = batch * batch_size;
      const size_t cur_batch_size = ((batch + 1) * batch_size <= csr.nnz())
                                        ? batch_size
                                        : csr.nnz() - batch * batch_size;

      double *const dx_tilde =
          &par_dx_tilde[tid * n_dimensions * 2 * batch_size];

      double *const dx = &par_dx[tid * n_dimensions * 2 * batch_size];

      int *const batch_rows = &coo.rowind()[batch_start_index];
      int *const batch_cols = &coo.colind()[batch_start_index];
      double *const batch_vals = &coo.val()[batch_start_index];

      std::set<size_t> seen_rows;
      for (size_t i = 0; i < cur_batch_size; ++i) {
        const size_t col = batch_cols[i];
        const size_t row = batch_rows[i];

        seen_rows.insert(row);
        seen_rows.insert(col);
      }

      // Add the full gradient in.
      size_t cur_itr = current_epoch++;
      for(const auto row:seen_rows){
        size_t itr_diff = 1;
        mutexes.at(row).lock();
        size_t c_row = counters[row];
        if(c_row < cur_itr){
          itr_diff = cur_itr-c_row;
          counters[row] = cur_itr;
        }
        mutexes.at(row).unlock();
        double *const x_row = &X[row * n_dimensions];
        for (size_t j = 0; j < n_dimensions; ++j) {
          x_row[j] += itr_diff*(dx_full[row* n_dimensions +j] / csr.nnz());
        }
      }

      common::compute_derivative(cur_batch_size, n_dimensions, eta, batch_rows,
                                batch_cols, batch_vals, X_tilde, dx_tilde);

      total_cost_reducer.update(tid,
          common::compute_derivative(cur_batch_size, n_dimensions, eta,
                                    batch_rows, batch_cols, batch_vals, X, dx));

      for (size_t j = 0; j < 2 * cur_batch_size * n_dimensions; ++j) {
        dx[j] -= dx_tilde[j];
      }

      for (size_t i = 0; i < cur_batch_size; ++i) {
        const size_t col = batch_cols[i];
        const size_t row = batch_rows[i];
        const double value = batch_vals[i];

        double *const x_row = &X[row * n_dimensions];
        double *const x_col = &X[col * n_dimensions];

        for (size_t j = 0; j < n_dimensions; ++j) {
          x_row[j] += (dx[2 * i * n_dimensions + j] / cur_batch_size);
          x_col[j] += (dx[(2 * i + 1) * n_dimensions + j] / cur_batch_size);
        }
      }
    });

    // Cleanup rows that need the full gradient added in.
    const size_t cur_itr = current_epoch++;

    par::for_range(0, coo.n(), [&](const size_t tid, const size_t row) {
      const size_t c_row = counters[row];
      size_t itr_diff = cur_itr - c_row;
      double *const x_row = &X[row * n_dimensions];
      for (size_t j = 0; j < n_dimensions; ++j) {
        x_row[j] += itr_diff * (dx_full[row * n_dimensions + j] / csr.nnz());
      }
    });

    total_error = total_cost_reducer.evaluate(0) / csr.nnz();
    if (save_epochs != 0 && (epoch + 1) % save_epochs == 0){
      X_procrustes = utils::orthogonal_procrustes(return_X);
    }
    utils::log_epoch_information(
        X_procrustes, vocab, output_file, epoch, n_epochs, save_epochs, log_epochs,
        total_error, tol, eta, itr_timer, [&]() {
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
  free(par_dx_tilde);
  free(dx_full);
  free(X_tilde);
  free(counters);

  ThreadPool::delete_thread_pool();


  return return_X;
}

}  // en
