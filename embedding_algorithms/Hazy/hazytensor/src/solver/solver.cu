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
#include "timer.h"
#include "utils.h"
#include <cuda_runtime_api.h>
#include <cuda.h>

namespace solver {
std::unique_ptr<double> gpu_power_iteration(
    const CSR<double> &cooccurrence, const size_t n_iterations,
    const size_t n_dimensions, const double tol, const size_t print_iterations,
    const std::string output_file, const std::vector<std::string> &vocab,
    const size_t seed) {

  const unsigned int N = 1048576;
    const unsigned int bytes = N * sizeof(int);
    int *h_a = (int*)malloc(bytes);
    int *d_a;
  cudaMalloc((int**)&d_a, bytes);

}

std::unique_ptr<double> gpu_sgd(
    const CSR<double> &cooccurrence, const COO<double> &coo,
    const size_t n_iterations, const size_t n_dimensions, const double tol,
    const size_t print_iterations, const std::string output_file,
    const std::vector<std::string> &vocab, const size_t seed) {

}

}  // en
