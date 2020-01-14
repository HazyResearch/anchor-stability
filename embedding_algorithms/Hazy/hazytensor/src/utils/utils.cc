#include "utils.h"
#include <cblas.h>
#include <lapacke.h>
#include <math.h>
#include <stdio.h>
#include <atomic>
#include <cstring>
#include <fstream>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <unordered_map>
#include "timer.h"

#ifndef __APPLE__
#include <omp.h>
#endif

namespace utils {

// On a mac we run single threaded due to the nastiness of getting openmp to
// work.
void *const get_parallel_buffer(const size_t n_threads,
                                const size_t buffer_size) {
  return malloc(n_threads * buffer_size);
}

std::vector<std::string> load_vocab(const std::string &filepath) {
  FILE *pFile;
  long lSize;
  char *buffer;
  size_t result;

  pFile = fopen(filepath.c_str(), "rb");
  if (pFile == NULL) {
    fputs("File error", stderr);
    exit(1);
  }

  // obtain file size:
  fseek(pFile, 0, SEEK_END);
  lSize = ftell(pFile);
  rewind(pFile);

  // allocate memory to contain the whole file:
  buffer = (char *)malloc(sizeof(char) * lSize);
  if (buffer == NULL) {
    fputs("Memory error", stderr);
    exit(2);
  }

  // copy the file into the buffer:
  result = fread(buffer, 1, lSize, pFile);
  if (result != lSize) {
    fputs("Reading error", stderr);
    exit(3);
  }

  std::vector<std::string> vocab;
  char *pch = strtok(buffer, " \n");
  while (pch != NULL) {
    vocab.push_back(pch);
    pch = strtok(NULL, " \n");

    pch = strtok(NULL, " \n");
  }

  // terminate
  fclose(pFile);
  free(buffer);
  return vocab;
}

void save_to_file(const DenseMatrix<double>& matrix,
                  const std::vector<std::string> &vocab,
                  const std::string &filename) {

  const double *const matrix_data = matrix.data();
  const size_t n_rows = matrix.n_rows();
  const size_t n_cols = matrix.n_cols();
  std::ofstream myfile;
  myfile.open(filename);
  for (size_t i = 0; i < n_rows; ++i) {
    myfile << vocab.at(i) << " ";
    for (size_t j = 0; j < n_cols; ++j) {
      myfile << matrix_data[i * n_cols + j] << " ";
    }
    myfile << "\n";
  }
}

DenseMatrix<double> initialize_embeddings(const size_t n_rows,
                                          const size_t n_cols,
                                          DenseMatrix<double> previous_model,
                                          DenseMatrix<double>& r,
                                          const size_t seed) {
  if (previous_model.is_valid()) {
    LOG("Using previous model!\n");
    return std::move(previous_model);
  } else {
    LOG("Randomly initializing embeddings!\n");
    DenseMatrix<double> embedding = DenseMatrix<double>(n_rows, n_cols);
    embedding.rand_uniform_init_no_par(seed);
//    embedding.qr(embedding, r);
    return embedding;
  }
}

DenseMatrix<double> load_pre_trained_model(const size_t m, const size_t n,
                                           const std::string &filepath,
                                           bool new_corpus,
                                           const std::vector<std::string> &new_vocab,
                                           const size_t seed) {
  if (filepath.empty())
    return DenseMatrix<double>();

  // Check if file exists
  FILE *pFile = fopen(filepath.c_str(), "rb");
  if (pFile == NULL) {
    fputs("Pretrained model error.", stderr);
    exit(1);
  }
  fclose(pFile);

  // Using pre-trained embedding from a different corpus
  if (new_corpus) {

    // Get number of words in old corpus
    int m_prev = 0;
    std::ifstream infile(filepath.c_str());
    std::string line;
    while (std::getline(infile, line))
      ++m_prev;
    infile.clear();
    infile.seekg(0, std::ios::beg);

    // Dimension n must be the same but number of words m_prev may be different than m
    double *prev_embedding = (double *)malloc(m_prev * n * sizeof(double));
    memset(prev_embedding, 0, m_prev * n * sizeof(double));

    size_t word_idx = 0;
    std::string word;
    double w;
    std::unordered_map<std::string,int> prev_vocab_map;

    while (word_idx != m_prev && std::getline(infile, line)) {
      std::istringstream iss(line);
      iss >> word;
      prev_vocab_map.insert({word, word_idx});
      for (size_t i = 0; i < n; ++i) {
        iss >> w;
        prev_embedding[word_idx * n + i] = w;
      }
      word_idx++;
    }
    infile.close();

    // Create new embedding as random matrix and fill with previous values in
    // appropriate locations
    double *embedding = (double *)malloc(m * n * sizeof(double));
    memset(embedding, 0, m * n * sizeof(double));

    // Uniform distribution
    std::default_random_engine generator;
    generator.seed(seed);
    std::normal_distribution<double> distribution(0, 1. / n);

    // Iterate through new vocab, if vocab overlaps, load previous word vector into new embedding
    for (int i = 0; i < m; i++){
      std::unordered_map<std::string,int>::const_iterator got = prev_vocab_map.find (new_vocab[i]);
      if (got != prev_vocab_map.end()) {
        for (int j = 0; j < n; ++j) {
          int prev_idx = got->second;
          embedding[i*n + j] = prev_embedding[prev_idx*n + j];
        }
      }
      else {
        for (int j = 0; j < n; ++j) {
          embedding[i*n + j] = distribution(generator);
        }
      }
    }
    LOG("Done loading pre-trained model!\n");
    // Free previous embedding
    free(prev_embedding);
    return DenseMatrix<double>(m,n,std::unique_ptr<double>(embedding));
  }

  // Load pre-trained embedding that uses same corpus as new embedding
  double *embedding = (double *)malloc(m * n * sizeof(double));
  memset(embedding, 0, m * n * sizeof(double));

  std::ifstream infile(filepath.c_str());
  std::string line;
  size_t word_idx = 0;
  std::string word;
  double w;

  while (word_idx != m && std::getline(infile, line)) {
    std::istringstream iss(line);
    iss >> word;
    for (size_t i = 0; i < n; ++i) {
      iss >> w;
      embedding[word_idx * n + i] = w;
    }
    word_idx++;
  }
  infile.close();
  LOG("Done loading pre-trained model!");
  return DenseMatrix<double>(m,n,std::unique_ptr<double>(embedding));
}

DenseMatrix<double> load_pre_trained_model(const size_t n,
                                          const std::string &filepath,
                                          std::vector<std::string> &vocab) {
  if (filepath.empty())
    return DenseMatrix<double>();

  // Check if file exists
  FILE *pFile = fopen(filepath.c_str(), "rb");
  if (pFile == NULL) {
    fputs("Pretrained model error.", stderr);
    exit(1);
  }
  fclose(pFile);

  int m = 0;
  std::ifstream infile(filepath.c_str());
  std::string line;
  while (std::getline(infile, line))
    ++m;
  infile.clear();
  infile.seekg(0, std::ios::beg);

  double *embedding = (double *)malloc(m * n * sizeof(double));
  memset(embedding, 0, m * n * sizeof(double));

  size_t word_idx = 0;
  std::string word;
  double w;
  while (word_idx != m && std::getline(infile, line)) {
    std::istringstream iss(line);
    iss >> word;
    vocab.push_back(word);
    for (size_t i = 0; i < n; ++i) {
      iss >> w;
      embedding[word_idx * n + i] = w;
    }
    word_idx++;
  }
  infile.close();
  LOG("Done loading pre-trained model!");
  return DenseMatrix<double>(m,n,std::unique_ptr<double>(embedding));
}

DenseMatrix<double> orthogonal_procrustes(const DenseMatrix<double>& matrix, const size_t seed) {
  // Get the shape of matrix
  size_t m = matrix.n_rows();
  size_t n = matrix.n_cols();

  // Generate reference matrix for orthogonal procrustes
  DenseMatrix<double> B = DenseMatrix<double>(m, n);
  B.rand_xavier_init_no_par(seed);

  DenseMatrix<double> C = DenseMatrix<double>(n, n);

  double alpha = 1.0;
  double beta = 0.0;

  // C = matrix^T * B
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, n, m, alpha, matrix.data(), n, B.data(), n, beta, C.data(), n);

  // U * S * VT = SVD(C)
  DenseMatrix<double> U = DenseMatrix<double>(n, n);
  DenseMatrix<double> S = DenseMatrix<double>(n, 1);
  DenseMatrix<double> VT = DenseMatrix<double>(n, n);
  DenseMatrix<double> superb = DenseMatrix<double>(n, 1);
  LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', n, n, C.data(), n, S.data(), U.data(), n, VT.data(), n, superb.data());

  // R = U * VT
  DenseMatrix<double> R = DenseMatrix<double>(n, n);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, alpha, U.data(), n, VT.data(), n, beta, R.data(), n);

  // After orthogonal procrustes, X should be X * R
  DenseMatrix<double> XR = DenseMatrix<double>(m, n);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, n, alpha, matrix.data(), n, R.data(), n, beta, XR.data(), n);

  return XR;
}

DenseMatrix<double> procrustes_pi(const DenseMatrix<double>& matrix) {
  // Get the shape of matrix
  size_t m = matrix.n_rows();
  size_t n = matrix.n_cols();

  // Generate orthogonal procrustes diagonal matrix for power method which has
  // the diagonal value as 1.0 or -1.0 based on the sign of first row of matrix
  DenseMatrix<double> B = DenseMatrix<double>(n, n);
  B.zero();
  double *B_ = B.data();

  for (size_t i = 0; i < n; ++i)
    if (matrix.data()[i] > 0)
      B_[i * n + i] = 1.0;
    else
      B_[i * n + i] = -1.0;

  DenseMatrix<double> C = DenseMatrix<double>(m, n);

  double alpha = 1.0;
  double beta = 0.0;

  // C = matrix * B
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, n, alpha, matrix.data(), n, B.data(), n, beta, C.data(), n);
  return C;
}

bool check_print_log_condition(const size_t cur_epoch,
                               const size_t log_epochs,
                               const size_t n_epochs) {
  if (log_epochs == 0) return false;
  if (cur_epoch == 0) return true;
  if (cur_epoch == n_epochs - 1) return true;
  if ((cur_epoch + 1) % log_epochs == 0) return true;
  return false;
}

}
