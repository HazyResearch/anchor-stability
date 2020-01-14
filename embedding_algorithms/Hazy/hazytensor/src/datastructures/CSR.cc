#include "CSR.h"
#include "utils/utils.h"

namespace {

template <class T>
void sort(int *col_idx, T *a, int start, int end) {
  int i, j, it;
  T dt;

  for (i = end - 1; i > start; i--)
    for (j = start; j < i; j++)
      if (col_idx[j] > col_idx[j + 1]) {
        if (a) {
          dt = a[j];
          a[j] = a[j + 1];
          a[j + 1] = dt;
        }
        it = col_idx[j];
        col_idx[j] = col_idx[j + 1];
        col_idx[j + 1] = it;
      }
}

template <class T>
void coo2csr(size_t n, size_t nz, T *a, int *i_idx, int *j_idx, T *csr_a,
             int *col_idx, size_t *row_start, bool sort_rows) {
  size_t i, l;

#pragma omp parallel for  // simd
  for (i = 0; i <= n; i++) row_start[i] = 0;

  /* determine row lengths */
  for (i = 0; i < nz; i++) row_start[i_idx[i] + 1]++;

  for (i = 0; i < n; i++) row_start[i + 1] += row_start[i];

  /* go through the structure  once more. Fill in output matrix. */
  for (l = 0; l < nz; l++) {
    i = row_start[i_idx[l]];
    csr_a[i] = a[l];
    col_idx[i] = j_idx[l];
    row_start[i_idx[l]]++;
  }

  /* shift back row_start */
  for (i = n; i > 0; i--) row_start[i] = row_start[i - 1];

  row_start[0] = 0;

  if (!sort_rows) return;

#pragma omp parallel for schedule(dynamic)
  for (i = 0; i < n; i++) {
    sort<T>(col_idx, csr_a, row_start[i], row_start[i + 1]);
  }
}
}

template <class T>
bool CSR<T>::find(const size_t i, const size_t j) const {
  if (i >= n_) return 0;
  // TODO: Make this a binary search.
  for (int rs = indptr_[i]; rs < indptr_[i + 1]; ++rs) {
    if (j == indices_[rs] && data_[rs] != 0) {
      return 1;
    }
  }
  return 0;
}

template <class T>
void CSR<T>::print() const {
  for (size_t i = 0; i < n_; ++i) {
    for (int rs = indptr_[i]; rs < indptr_[i + 1]; ++rs) {
      std::cout << "(" << i << " " << indices_[rs] << ")\t" << data_[rs]
                << std::endl;
    }
  }
}

template <class T>
CSR<T> CSR<T>::from_coo(size_t _n, size_t _nnz, int *_i, int *_j, T *_data,
                        bool sort_rows) {
  T *data_ = (T *)malloc(sizeof(T) * _nnz);
  int *indices_ = (int *)malloc(sizeof(int) * _nnz);
  size_t *indptr_ = (size_t *)malloc(sizeof(size_t) * (_n + 1));

  coo2csr<T>(_n, _nnz, _data, _i, _j, data_, indices_, indptr_, sort_rows);
  return CSR<T>(_n, _nnz, indptr_, indices_, data_);
}

template <>
void CSR<double>::ppmi() {
  double D = 0.0;
  size_t n = n_;
  size_t nnz = nnz_;
  double *wc = (double *)malloc(n * sizeof(double));
  double *wc0 = (double *)malloc(nnz * sizeof(double));

#pragma omp parallel for reduction(+ : D)
  for (size_t row = 0; row < n; ++row) {
    double sum = 0.0;
    for (size_t colidx = indptr_[row]; colidx < indptr_[row + 1]; ++colidx) {
      sum += data_[colidx];
    }
    D += sum;
    wc[row] = sum;
  }

  double *wc1 = (double *)malloc(nnz * sizeof(double));
  double *D_vals = (double *)malloc(nnz * sizeof(double));

#pragma omp parallel for schedule(dynamic)
  for (size_t row = 0; row < n; ++row) {
    for (size_t colidx = indptr_[row]; colidx < indptr_[row + 1]; ++colidx) {
      wc1[colidx] = wc[indices_[colidx]];
      wc0[colidx] = wc[row];
      D_vals[colidx] = D;
    }
  }

// v = torch.log(v) + torch.log(torch.DoubleTensor(nnz).fill_(D)) -
// torch.log(wc0) - torch.log(wc1)
// clamp(min=0)
#pragma omp parallel for  // simd
  for (size_t i = 0; i < nnz; ++i) {
    data_[i] = std::max(
        (log(data_[i]) + log(D_vals[i]) - log(wc0[i]) - log(wc1[i])), 0.0);
  }

  delete wc;
  delete wc0;
  delete wc1;
  delete D_vals;
}

template class CSR<float>;
template class CSR<double>;
