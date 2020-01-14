//  Linear algebra multiplication operations
#include "multiply.h"
#include "utils/utils.h"

namespace ops {

template <>
void multiply(DenseVector<double>& out_vector, const CSR<double>& in_matrix,
              const DenseVector<double>& in_vector) {
  par::for_range(0, in_matrix.n(), 100,
                 [&](const size_t tid, const size_t row) {
                   double val = 0.0;
                   for (size_t colidx = in_matrix.indptr()[row];
                        colidx < in_matrix.indptr()[row + 1]; ++colidx) {
                     val += (in_vector.data()[in_matrix.indices()[colidx]] *
                             in_matrix.data()[colidx]);
                   }
                   out_vector.data()[row] = val;
                 });
}

}  // end namespace ops
