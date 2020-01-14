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
#include "ops/multiply.h"

namespace {

double compute_loss(const COO<double> &coo, double *const X,
                    double *const lambda, const size_t n_dimensions) {
  auto total_cost_reducer = par::reducer<double>(
      0, [](const double a, const double b) { return a + b; });
  par::for_range(0, coo.nnz(), [&](const size_t tid, const size_t i) {
    const int row = coo.rowind()[i];
    const int col = coo.colind()[i];
    const double value = coo.val()[i];

    const double *const x_row = &X[row * n_dimensions];
    const double *const x_col = &X[col * n_dimensions];
    double pred = 0.0;
    for (size_t j = 0; j < n_dimensions; ++j) {
      pred += x_row[j] * lambda[j * n_dimensions + j] * x_col[j];
    }

    const double error = pred - value;
    total_cost_reducer.update(tid,(error * error));
  });
  return total_cost_reducer.evaluate(0) / coo.nnz();
}

}  // end namespace

namespace solver {
DenseMatrix<double> cpu_simultaneous_power_iteration(
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
    const size_t n_threads,
    DenseMatrix<double> previous_model,
    size_t& used_epochs,
    bool debug) {
  const auto solver_time = timer::start_clock();

  ThreadPool::initialize_thread_pool(n_threads);
  srand(seed);

  // Buffer for the r from qr decomposition.
  DenseMatrix<double> r(n_dimensions, n_dimensions);

  // Initialize embeddings
  DenseMatrix<double> embedding = utils::initialize_embeddings(
      csr.n(), n_dimensions, std::move(previous_model), r, seed);

  // Temporary buffer for swapping out values in computation.
  DenseMatrix<double> tmp_embedding(csr.n(), n_dimensions);

  // Current epochs eigen values (for convergence check).
  double *cur_eigs = (double *)malloc(n_dimensions * sizeof(double));
  // Previous epochs eigen values (for convergence check). Init to 0.
  double *last_eigs = (double *)malloc(n_dimensions * sizeof(double));
  memset(last_eigs, 0, n_dimensions * sizeof(double));

  // Kernel.
  size_t epoch = 0;
  double norm = 0.0;
  double loss = 0.0;
  DenseMatrix<double> embedding_procrustes;

  for (; epoch < n_epochs; ++epoch) {
    const auto epoch_timer = timer::start_clock();

    // Swap buffers
    std::swap(tmp_embedding, embedding);

    // Zero out embeddings for AxPy add.
    embedding.zero();

    // embedding = coocurrence*tmp_embedding.
    // This is a fast way to do a sparse-matrix/dense-matrix multiplication.
    par::for_range(0, csr.n(), 100, [&](const size_t tid, const size_t row){
      for (size_t colidx = csr.indptr()[row];
           colidx < csr.indptr()[row + 1]; ++colidx) {
        cblas_daxpy(
            n_dimensions, csr.data()[colidx],
            &tmp_embedding.data()[csr.indices()[colidx] * n_dimensions], 1,
            &embedding.data()[row * n_dimensions], 1);
      }
    });

    // cur_eigs = norm(embedding)
    par::for_range(0, n_dimensions, [&](const size_t tid, const size_t i){
      cur_eigs[i] = cblas_dnrm2(csr.n(), &embedding.data()[i], n_dimensions);
    });
    std::sort(cur_eigs, cur_eigs + n_dimensions);

    // Print eigenvalues
    if (log_epochs != 0 && (epoch + 1) % log_epochs == 0 && debug) {
      for(int e = 0; e < n_dimensions; e++){
        LOG(cur_eigs[e]);
      }
    }

    // embedding, r = qr(embedding)
    embedding.qr(embedding, r);

    // vecnorm(last_eigs-cur_eigs)
    auto norm_reducer = par::reducer<double>(
        0, [](const double a, const double b) { return a + b; });
    par::for_range(0, n_dimensions, [&](const size_t tid, const size_t i){
      const double diff = last_eigs[i] - cur_eigs[i];
      norm_reducer.update(tid, (diff*diff));
    });
    norm = sqrt(norm_reducer.evaluate(0) / n_dimensions);
    std::swap(cur_eigs, last_eigs);

    // Only perform procrustes if saving the embedding this epoch
    if (save_epochs != 0 && (epoch + 1) % save_epochs == 0){
      embedding_procrustes = utils::procrustes_pi(embedding);
    }
    utils::log_epoch_information(
        embedding_procrustes, vocab, output_file, epoch, n_epochs, save_epochs, log_epochs,
        norm, tol, -1, epoch_timer, [&]() {
          return compute_loss(coo, embedding.data(), r.data(), n_dimensions);
        });

    // Convergence check!
    if (norm <= tol) {
      LOG("Convergence condition met. Returning!");
      break;
    }
  }

  timer::stop_clock("SOLVER TIME", solver_time);

  // Procrustes
  const auto procrustes_pi = timer::start_clock();
  embedding_procrustes = utils::procrustes_pi(embedding);
  timer::stop_clock("PROCRUSTES TIME", procrustes_pi);

  const auto save_to_file = timer::start_clock();
  if (epoch == n_epochs){
    // Already incremented
    utils::save_to_file(embedding, vocab,
                      output_file + "." +
                          std::to_string(epoch) + ".final_orig");
    utils::save_to_file(embedding_procrustes, vocab,
                      output_file + "." +
                          std::to_string(epoch) + ".final");
  }
  else {
    utils::save_to_file(embedding, vocab,
                      output_file + "." +
                          std::to_string(epoch+1) + ".final_orig");
    utils::save_to_file(embedding_procrustes, vocab,
                        output_file + "." +
                            std::to_string(epoch+1) + ".final");
  }
  timer::stop_clock("WRITING FILE", save_to_file);

  used_epochs += epoch;

  ThreadPool::delete_thread_pool();
  free(cur_eigs);
  free(last_eigs);
  return embedding_procrustes;
}

DenseMatrix<double> cpu_deflation_power_iteration(
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
    const size_t n_threads,
    DenseMatrix<double> previous_model,
    size_t& used_epochs) {
  const auto solver_time = timer::start_clock();

  ThreadPool::initialize_thread_pool(n_threads);
  srand(seed);

  // Buffer for the r from qr decomposition.
  DenseMatrix<double> r (n_dimensions, n_dimensions);

  // Initialize embeddings
  DenseMatrix<double> xs = utils::initialize_embeddings(
      csr.n(), n_dimensions, std::move(previous_model), r,
      seed);

  DenseVector<double> eigen_values(n_dimensions);
  eigen_values.zero();

  DenseVector<double> eigen_vector(csr.n());

  // Since we calculate row by row, xs transpose will be faster
  DenseMatrix<double> xs_t = xs.transpose();

  double norm = 0;


  // Calculate eigenvectors one by one
  for(size_t j = 0; j < n_dimensions; ++j){
    LOG("Computing eigenvector " + std::to_string(j));
    // Compute the j_{th} eigenvector
    DenseVector<double> x = xs_t.get_row(j);
    // Nomralization
    x.normalize();

    double diff = 0.0;

    // Power iteration
    size_t epoch = 0;
    for (; epoch < n_epochs; ++epoch) {
      // x = A * x
      ops::multiply(eigen_vector, csr, x);
      // Deflation
      // A' = A - \sum_0^{j-1} eigenvalue_k * eigenvector_k * eigenvector_k^T
      if (j>0) {
        DenseVector<double> ts (j);
        par::for_range(0, j, [&](const size_t tid, const size_t k) {
          const double *const x_k = &xs_t.data()[k * csr.n()];
          ts.data()[k] = cblas_ddot(csr.n(), x_k, 1, eigen_vector.data(), 1);
        });
        for (size_t k = 0; k < j; ++k) {
          double t = ts.data()[k];
          for (size_t i = 0; i < csr.n(); ++i) {
            eigen_vector.data()[i] -= t * xs_t.data()[k * csr.n() + i];
          }
        }
      }
      double eig = eigen_vector.norm();
      eigen_vector.normalize();
      std::memcpy(&xs_t.data()[j * csr.n()], eigen_vector.data(), sizeof(double) * csr.n());
      diff = eig - eigen_values.data()[j];
      eigen_values.data()[j] = eig;
      if (std::abs(diff) < tol) {
        LOG("Convergence condition met at epoch " + std::to_string(epoch) + " Returning!");
        break;
      }
    }
    if (epoch == n_epochs)
      LOG("Compute finished at epoch " + std::to_string(n_epochs));
    norm += diff * diff;
  }

  eigen_values.print();
  xs = xs_t.transpose();

  timer::stop_clock("SOLVER TIME", solver_time);

  // Procrustes
  const auto procrustes_pi = timer::start_clock();
  DenseMatrix<double> xs_procrustes = utils::procrustes_pi(xs);
  timer::stop_clock("PROCRUSTES TIME", procrustes_pi);

  const auto save_to_file = timer::start_clock();
  utils::save_to_file(xs, vocab,
                      output_file +  "." +
                          std::to_string(n_epochs) + ".final_orig");
  utils::save_to_file(xs_procrustes, vocab,
                      output_file +  "." +
                          std::to_string(n_epochs) + ".final");
  timer::stop_clock("WRITING FILE", save_to_file);

  used_epochs += n_epochs;

  ThreadPool::delete_thread_pool();
  return xs;
}

}  // en
