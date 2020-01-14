import numpy as np
import os
import scipy.sparse
from scipy.sparse import coo_matrix, csr_matrix
import unittest

import hazy
import hazytensor


class MainTest(unittest.TestCase):
    def test_coo_sample(self):
        n_words = 10
        percent = 10
        seed = 1234
        symmetric = True

        # Generate synthetic data for original coo
        x = scipy.sparse.random(
            n_words,
            n_words,
            density=0.6,
            format="coo",
            data_rvs=lambda s: np.random.rand(s),
        )
        mat = x * x.T
        row, col, data = [], [], []
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                row.append(i)
                col.append(j)
                data.append(mat[i, j])
        row = np.array(row)
        col = np.array(col)
        data = np.array(data)
        sm = coo_matrix((data, (row, col)))
        hazy_csr = hazytensor.DoubleCSR.from_coo(sm.row, sm.col, sm.data.astype(np.float64))
        coo = hazytensor.DoubleCOO.from_csr(hazy_csr)

        # sample coo
        sampled_coo = hazytensor.DoubleCOO.sample(coo, percent, seed, symmetric).scipy()

        # check density
        self.assertEqual(int(sm.nnz*percent/100.), sampled_coo.nnz)

        # compare entries of coo and sampled coo
        data = {}
        for i, j, v in zip(sm.row, sm.col, sm.data):
            data[(i, j)] = v

        for i, j, v in zip(sampled_coo.row, sampled_coo.col, sampled_coo.data):
            self.assertEqual(data[(i, j)], v)


if __name__ == "__main__":
    unittest.main()