//  Linear algebra multiplication operations

#ifndef MULTIPLY_H
#define MULTIPLY_H

#include <memory>
#include <vector>
#include "datastructures/COO.h"
#include "datastructures/CSR.h"
#include "datastructures/DenseVector.h"
#include "datastructures/DenseMatrix.h"

namespace ops {

template <class T>
void multiply(DenseVector<T>& out_vector, const CSR<T>& in_matrix,
              const DenseVector<T>& in_vector);

}  // end namespace ops

#endif
