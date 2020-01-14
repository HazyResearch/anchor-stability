#ifndef SOLVER_H
#define SOLVER_H

#include <memory>
#include <vector>
#include <set>
#include "datastructures/COO.h"
#include "datastructures/CSR.h"
#include "datastructures/DenseVector.h"
#include "datastructures/DenseMatrix.h"

namespace solver {

std::unique_ptr<double> gpu_power_iteration(
    const CSR<double> &csr,
    const size_t n_epochs,
    const size_t n_dimensions,
    const double tol,
    const size_t save_epochs,
    const size_t log_epochs,
    const std::string output_file,
    const std::vector<std::string> &vocab,
    const size_t seed);

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
    bool debug);

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
    size_t& used_epochs);

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
    size_t& used_epochs);

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
    const size_t batch_size,
    const double lr,
    const double beta,
    const double lambda,
    const double lr_decay,
    const size_t n_threads,
    const std::set<size_t> freeze_idx_set,
    const size_t freeze_n_epochs,
    DenseMatrix<double> previous_model,
    size_t& used_epochs);

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
    const size_t batch_size,
    const double lr,
    const size_t n_threads,
    DenseMatrix<double> previous_model,
    size_t& used_epochs);

DenseMatrix<double> cpu_hybird_pi_sgd(
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
    const size_t batch_size,
    const double lr,
    const double beta,
    const double lambda,
    const double lr_decay,
    const size_t n_threads,
    const std::set<size_t> freeze_idx_set,
    const size_t freeze_n_epochs,
    DenseMatrix<double> previous_model);
}  // end namespace solver

#endif
