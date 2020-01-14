#ifndef PYSOLVER_H
#define PYSOLVER_H

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stddef.h>
#include <set>

#include "PyCSR.h"
#include "datastructures/DenseMatrix.h"
#include "solver/solver.h"
#include "utils/utils.h"

namespace py = pybind11;

namespace PySolver {
void solve(const PyCSR<double> &pycsr, const PyCOO<double> &pycoo,
           const size_t n_epochs, const size_t n_dimensions,
           const double tol, const size_t save_epochs,
           const size_t log_epochs, const std::string output_file,
           const std::string vocab_file, const std::string solver,
           const size_t seed, const size_t svrg_freq, const size_t batch_size,
           const double lr, const double beta, const double lambda,
           const double lr_decay,const size_t n_threads,
           const std::string freeze_vocab_filename,
           const size_t freeze_n_epochs,
           const std::string pre_trained,
           bool new_corpus) {
  const std::vector<std::string> vocab = utils::load_vocab(vocab_file);

  std::set<size_t> freeze_idx_set;
  if (freeze_vocab_filename != "") {
    const std::vector<std::string> freeze_vocab = utils::load_vocab(freeze_vocab_filename);
    std::set<std::string> freeze_vocab_set;
    for (size_t i = 0; i < freeze_vocab.size(); ++i)
      freeze_vocab_set.insert(freeze_vocab.at(i));

    for (size_t i = 0; i < vocab.size(); ++i) {
      std::set<std::string>::iterator it = freeze_vocab_set.find(vocab.at(i));
      if(it != freeze_vocab_set.end())
        freeze_idx_set.insert(i);
    }
  }


  DenseMatrix<double> pre_trained_model =
      utils::load_pre_trained_model(pycsr.csr.n(), n_dimensions, pre_trained, new_corpus, vocab, seed);
  size_t used_epochs = 0;
  if (solver == "pi") {
    solver::cpu_simultaneous_power_iteration(
        pycsr.csr,
        pycoo.coo,
        n_epochs,
        n_dimensions,
        tol,
        save_epochs,
        log_epochs,
        output_file,
        vocab,
        seed,
        n_threads,
        std::move(pre_trained_model),
        used_epochs,
        false /* debug */);
  } else if (solver == "dpi") {
    solver::cpu_deflation_power_iteration(
        pycsr.csr,
        pycoo.coo,
        n_epochs,
        n_dimensions,
        tol,
        save_epochs,
        log_epochs,
        output_file,
        vocab,
        seed,
        n_threads,
        std::move(pre_trained_model),
        used_epochs);
  } else if (solver == "gd") {
    solver::cpu_gd(
        pycsr.csr,
        pycoo.coo,
        n_epochs,
        n_dimensions,
        tol,
        save_epochs,
        log_epochs,
        output_file,
        vocab,
        seed,
        lr,
        n_threads,
        std::move(pre_trained_model),
        used_epochs);
  } else if (solver == "sgd") {
    solver::cpu_sgd(
        pycsr.csr,
        pycoo.coo,
        n_epochs,
        n_dimensions,
        tol,
        save_epochs,
        log_epochs,
        output_file,
        vocab,
        seed,
        batch_size,
        lr,
        beta,
        lambda,
        lr_decay,
        n_threads,
        freeze_idx_set,
        freeze_n_epochs,
        std::move(pre_trained_model),
        used_epochs);
  } else if (solver == "svrg") {
    solver::cpu_svrg(
        pycsr.csr,
        pycoo.coo,
        n_epochs,
        n_dimensions,
        tol,
        svrg_freq,
        save_epochs,
        log_epochs,
        output_file,
        vocab,
        seed,
        batch_size,
        lr,
        n_threads,
        std::move(pre_trained_model),
        used_epochs);
  } else if (solver == "hybird_pi_sgd") {
    solver::cpu_hybird_pi_sgd(
        pycsr.csr,
        pycoo.coo,
        n_epochs,
        n_dimensions,
        tol,
        save_epochs,
        log_epochs,
        output_file,
        vocab,
        seed,
        batch_size,
        lr,
        beta,
        lambda,
        lr_decay,
        n_threads,
        freeze_idx_set,
        freeze_n_epochs,
        std::move(pre_trained_model));
  } else {
    std::cout << "Solver " << solver << " not supported!"
              << " Use either 'pi', 'dpi', 'gd', 'sgd', 'svrg', 'hybird_pi_sgd'."
              << std::endl;
    exit(1);
  }
}
};

#endif
