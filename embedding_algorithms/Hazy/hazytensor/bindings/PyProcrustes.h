#ifndef PYPROCRUSTES_H
#define PYPROCRUSTES_H

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stddef.h>

#include "PyDenseMatrix.h"
#include "datastructures/DenseMatrix.h"
#include "utils/utils.h"

namespace py = pybind11;

namespace PyProcrustes {

    PyDenseMatrix<double> orthogonal_procrustes(const PyDenseMatrix<double> &pydensematrix, const size_t seed) {
        DenseMatrix<double> XR = utils::orthogonal_procrustes(
            pydensematrix.dense_matrix,
            seed);
        return PyDenseMatrix<double>(XR);
    }

    PyDenseMatrix<double> procrustes_pi(const PyDenseMatrix<double> &pydensematrix) {
        DenseMatrix<double> XR = utils::procrustes_pi(pydensematrix.dense_matrix);
        return PyDenseMatrix<double>(XR);
    }

};
#endif
