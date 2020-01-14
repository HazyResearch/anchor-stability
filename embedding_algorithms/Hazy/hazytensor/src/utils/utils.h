//  A collection of utility functions.

#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <string.h>
#include <vector>
#include "datastructures/COO.h"
#include "datastructures/DenseMatrix.h"
#include "timer.h"
#include "parallel.h"
#include "thread_pool.h"

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#ifdef DEBUG
#define LOG(x)                                                     \
  do {                                                             \
    std::cerr << "LOG(line: " << __LINE__ << " file: " << __FILENAME__ \
              << "): " << x << std::endl;                          \
  } while (0)

#else
#define LOG(x)
#endif

#define ASSERT(condition, message)                                           \
  {                                                                          \
    if (!(condition)) {                                                      \
      std::cerr << "ASSERTION ERROR\nmessage: " << message                   \
                << " condition: " << #condition << " @ " << __FILENAME__ << " (" \
                << __LINE__ << ")" << std::endl;                             \
    }                                                                        \
  }

namespace utils {

std::vector<std::string> load_vocab(const std::string &filepath);

//  Check whether we should print out information for the current epoch or not.
bool check_print_log_condition(const size_t cur_epoch, const size_t log_epochs,
                               const size_t n_epochs);

//  Returns the current thread id.
int get_tid();

//  Returns a buffer of size = buffer_size * n_threads;
void *const get_parallel_buffer(const size_t n_threads,
                                const size_t buffer_size);


//  Saves a word embedding matrix to disk. The file format is:
//      <string_vocab_word> <emb_matrix[0][0]>  <emb_vec[0][1]> ...
//      <string_vocab_word> <emb_matrix[1][0]>  <emb_vec[1][1]> ...
//      ...
void save_to_file(const DenseMatrix<double>& matrix,
                  const std::vector<std::string> &vocab,
                  const std::string &filename);

//  Return a previously trained model from disk or null if no filename
//  specified. The file format for this method is the same as the 'save_to_file'
//  method above.
DenseMatrix<double> load_pre_trained_model(const size_t m, const size_t n,
                                               const std::string &filepath,
                                               bool new_corpus,
                                               const std::vector<std::string> &new_vocab,
                                               const size_t seed);

//  Return a previously trained model from disk or null if no filename
//  specified. The file format for this method is the same as the 'save_to_file'
//  method above.
DenseMatrix<double> load_pre_trained_model(const size_t n,
                                               const std::string &filepath,
                                               std::vector<std::string> &vocab);

//  Returns the previous model if it is valid, otherwise allocates a dense
//  matrix, randomly initializes it to 0 or 1 values, and runs QR decomp.
DenseMatrix<double> initialize_embeddings(const size_t n_rows,
                                          const size_t n_cols,
                                          DenseMatrix<double> previous_model,
                                          DenseMatrix<double>& r,
                                          const size_t seed);

//  Applies orthogonal procrustes to matrix.
DenseMatrix<double> orthogonal_procrustes(const DenseMatrix<double> &matrix, const size_t seed=1234);

//  Applies procrustes to matrix for power method which makes the
//  first the dim of each egienvector positive.
DenseMatrix<double> procrustes_pi(const DenseMatrix<double> &matrix);

//  Logs information about the current epoch to the user and saves the
//  embeddings to disk.
template <typename F>
void log_epoch_information(
    const DenseMatrix<double>& embedding, const std::vector<std::string> &vocab,
    const std::string &output_file, const size_t cur_epoch,
    const size_t n_epochs, const size_t save_epochs, const size_t log_epochs,
    const double norm, const double tol, const double eta,
    const std::chrono::time_point<std::chrono::system_clock> epoch_timer,
    F loss_function) {
  if (log_epochs == 0) return;

  //  LOG information to the user.
  if (cur_epoch == 0 || cur_epoch == n_epochs - 1 ||
      (cur_epoch + 1) % log_epochs == 0 || norm < tol) {
    // Timing
    LOG("Time[EPOCH]: " + std::to_string(timer::stop_clock(epoch_timer)));
    // Calcualte the loss based on the observed matrix
    const double loss = loss_function();
    if (eta >= 0.0) {
      LOG("Learning rate at this epoch: " + std::to_string(eta));
    }
    LOG("Total loss at this epoch: " + std::to_string(loss));
    LOG("Epoch: " + std::to_string(cur_epoch+1) + " c: " + std::to_string(norm));
  }

  //  Save embeddings to disk.
  if (save_epochs != 0 &&
      (cur_epoch+1) % save_epochs == 0) {
    const auto save_to_file = timer::start_clock();
    if (cur_epoch == n_epochs){
      utils::save_to_file(embedding, vocab,
                        output_file + "." +
                            std::to_string(cur_epoch) + ".txt");
    }
    else {
      utils::save_to_file(embedding, vocab,
                          output_file + "." +
                              std::to_string(cur_epoch+1) + ".txt");
    }
    timer::stop_clock("WRITING FILE", save_to_file);
  }
}

}

#endif