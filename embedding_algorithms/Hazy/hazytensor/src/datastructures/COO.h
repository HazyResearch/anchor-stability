#ifndef COO_H
#define COO_H

#include <memory>
#include <stdio.h>
#include <string>
#include <iostream>
#include <climits>
#include <assert.h>
#include "CSR.h"

// Elements from the binary COO file read on disk.
template<class T>
struct COOElem {
  int row;
  int col;
  T val;
};

// Coordinate matrix.
template <class T>
class COO {
 public:
  COO(const size_t nnz, const size_t n, std::unique_ptr<int> rowind,
      std::unique_ptr<int> colind, std::unique_ptr<T> val)
      : nnz_(nnz),
        n_(n),
        rowind_(std::move(rowind)),
        colind_(std::move(colind)),
        val_(std::move(val)) {}

  // Constructs a COO from an array of COOElem.
  COO(const size_t num_bytes, COOElem<T>* buffer);

  void shuffle_inplace(const size_t seed);

  void to_file(const std::string& filepath);

  // Prints out the first N elements of the COO matrix (debug).
  // Note: only prints exact amount for even numbers (otherwise off by 1).
  void print(const size_t N = ULONG_MAX);

  static COO<T> from_file(const std::string& filepath);

  static COO<T> from_csr(const CSR<T>& csr);

  COO<T> sample(const size_t sample_percent, const size_t seed,
    const bool symmetric);

  inline size_t nnz() const {return nnz_;}
  inline size_t n() const {return n_;}
  inline int* rowind() const {return rowind_.get();}
  inline int* colind() const {return colind_.get();}
  inline T* val() const {return val_.get();}

 private:
  size_t nnz_;
  size_t n_;  // number of rows
  std::unique_ptr<int> rowind_;
  std::unique_ptr<int> colind_;
  std::unique_ptr<T> val_;
};

#endif
