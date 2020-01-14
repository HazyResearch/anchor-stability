#include "COO.h"
#include <algorithm>
#include <random>
#include "utils/utils.h"

template<class T>
COO<T>::COO(const size_t num_bytes, COOElem<T>* buffer)
    : nnz_(num_bytes / sizeof(COOElem<T>)) {
  rowind_ = std::unique_ptr<int>((int*)malloc(nnz_ * sizeof(int)));
  colind_ = std::unique_ptr<int>((int*)malloc(nnz_ * sizeof(int)));
  val_ = std::unique_ptr<T>((T*)malloc(nnz_ * sizeof(T)));

  size_t _n = 0;
  //#pragma omp parallel for reduction(max : _n)
  for (size_t i = 0; i < nnz_; ++i) {
    rowind_.get()[i] =
        (int)(buffer[i].row - 1);  // Convert 1-based to 0-based index.
    _n = (rowind_.get()[i] >= _n) ? rowind_.get()[i] : _n;
    colind_.get()[i] =
        (int)(buffer[i].col - 1);  // Convert 1-based to 0-based index.
    _n = (colind_.get()[i] >= _n) ? colind_.get()[i] : _n;
    val_.get()[i] = buffer[i].val;
  }
  n_ = _n + 1;
}

// Prints out the first N elements of the COO matrix (debug).
// Note: only prints exact amount for even numbers (otherwise off by 1).
template <class T>
void COO<T>::print(size_t N) {
  bool all_elems = (N == ULONG_MAX);
  N = (all_elems) ? nnz_ : N;
  for (size_t i = 0; i < N / 2; ++i) {
    std::cout << rowind_.get()[i] << " " << colind_.get()[i] << " "
              << val_.get()[i] << std::endl;
  }
  if (!all_elems) std::cout << "..." << std::endl;
  for (size_t i = nnz_ - (N / 2); i < nnz_; ++i) {
    std::cout << rowind_.get()[i] << " " << colind_.get()[i] << " "
              << val_.get()[i] << std::endl;
  }
}

template <>
void COO<double>::to_file(const std::string& filepath) {
  FILE* pFile;
  size_t result;

  pFile = fopen(filepath.c_str(), "wb");
  if (pFile == NULL) {
    fputs("File error", stderr);
    exit(1);
  }

  for (size_t i = 0; i < nnz_; ++i) {
    const int row =
        rowind_.get()[i] + 1;  // File format is 1-index based (GloVe).
    const int col =
        colind_.get()[i] + 1;  // File format is 1-index based (GloVe).
    const double value = val_.get()[i];

    result = fwrite(&row, sizeof(row), 1, pFile);
    if (result != 1) {
      fputs("Writing error", stderr);
      exit(3);
    }

    result = fwrite(&col, sizeof(col), 1, pFile);
    if (result != 1) {
      fputs("Writing error", stderr);
      exit(3);
    }

    result = fwrite(&value, sizeof(double), 1, pFile);
    if (result != 1) {
      fputs("Writing error", stderr);
      exit(3);
    }
  }

  fclose(pFile);
  LOG("COO written to binary file.");
}

template <>
COO<double> COO<double>::from_file(const std::string& filepath) {
  FILE* pFile;
  long lSize;
  char* buffer;
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
  buffer = (char*)malloc(sizeof(char) * lSize);
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

  COO<double> embedding = COO<double>(lSize, (COOElem<double>*)buffer);

  // terminate
  fclose(pFile);
  free(buffer);
  return embedding;
}

template <class T>
COO<T> COO<T>::from_csr(const CSR<T>& csr) {
  int* c_i = (int*)malloc(sizeof(int) * csr.nnz());
  int* c_j = (int*)malloc(sizeof(int) * csr.nnz());
  T* c_v = (T*)malloc(sizeof(T) * csr.nnz());

  size_t index = 0;
  for (size_t i = 0; i < csr.n(); ++i) {
    for (int rs = csr.indptr()[i]; rs < csr.indptr()[i + 1]; ++rs) {
      c_i[index] = i;
      c_j[index] = csr.indices()[rs];
      c_v[index] = csr.data()[rs];
      index++;
    }
  }
  return std::move(COO<T>(csr.nnz(), csr.n(), std::unique_ptr<int>(c_i),
                std::unique_ptr<int>(c_j), std::unique_ptr<T>(c_v)));
}

template <class T>
COO<T> COO<T>::sample(const size_t sample_percent, const size_t seed, bool symmetric) {
  int sample_size = nnz_ * sample_percent/100.;

  int unique_samples = symmetric ? sample_size / 2 : sample_size;

  std::unique_ptr<int> new_rowind_ =
      std::unique_ptr<int>((int*)malloc(sample_size * sizeof(int)));
  std::unique_ptr<int> new_colind_ =
      std::unique_ptr<int>((int*)malloc(sample_size * sizeof(int)));
  std::unique_ptr<T> new_val_ = std::unique_ptr<T>((T*)malloc(sample_size * sizeof(T)));

  size_t* indexes = (size_t*)malloc(nnz_ * sizeof(size_t));
  #pragma omp parallel for
  for (size_t i = 0; i < nnz_; ++i) {
    indexes[i] = i;
  }

  shuffle(indexes, indexes + nnz_, std::default_random_engine(seed));

  #pragma omp parallel for
  for (size_t i = 0; i < unique_samples; ++i) {
    new_rowind_.get()[i] = rowind_.get()[indexes[i]];
    new_colind_.get()[i] = colind_.get()[indexes[i]];
    new_val_.get()[i] = val_.get()[indexes[i]];
  }

  if (symmetric) {
    for (size_t i = 0; i < unique_samples; ++i) {
      new_rowind_.get()[i + unique_samples] = colind_.get()[indexes[i]];
      new_colind_.get()[i + unique_samples] = rowind_.get()[indexes[i]];
      new_val_.get()[i + unique_samples] = val_.get()[indexes[i]];
    }
  }

  COO<T> sampled_coo = COO<T>(sample_size, n_, std::move(new_rowind_),
    std::move(new_colind_), std::move(new_val_));
  return sampled_coo;
}

template<class T>
void COO<T>::shuffle_inplace(const size_t seed){
  std::unique_ptr<int> new_rowind_ =
      std::unique_ptr<int>((int*)malloc(nnz_ * sizeof(int)));
  std::unique_ptr<int> new_colind_ =
      std::unique_ptr<int>((int*)malloc(nnz_ * sizeof(int)));
  std::unique_ptr<T> new_val_ = std::unique_ptr<T>((T*)malloc(nnz_ * sizeof(T)));

  size_t* indexes = (size_t*)malloc(nnz_ * sizeof(size_t));

  #pragma omp parallel for
  for (size_t i = 0; i < nnz_; ++i) {
    indexes[i] = i;
  }

  shuffle(indexes, indexes + nnz_, std::default_random_engine(seed));

  #pragma omp parallel for
  for (size_t i = 0; i < nnz_; ++i) {
    new_rowind_.get()[i] = rowind_.get()[indexes[i]];
    new_colind_.get()[i] = colind_.get()[indexes[i]];
    new_val_.get()[i] = val_.get()[indexes[i]];
  }

  rowind_ = std::move(new_rowind_);
  colind_ = std::move(new_colind_);
  val_ = std::move(new_val_);

  delete indexes;
}

template class COO<float>;
template class COO<double>;
