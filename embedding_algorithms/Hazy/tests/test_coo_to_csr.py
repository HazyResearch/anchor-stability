import hashlib
import os
import unittest

import hazy
import numpy as np

import hazytensor

# Test coo to csr function


class MainTest(unittest.TestCase):
    def test_coo_to_csr(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        n_words = 10
        # Loads the output cooccur from GloVe into a scipy COO.

        data = {}

        coo = hazy.coo_from_file(dir_path + "/data/sample_data_1.bin").scipy()
        for i, j, v in zip(coo.row, coo.col, coo.data):
            data[(i, j)] = v

        print("Transforming from COO to CSR...")
        csr = hazytensor.DoubleCSR.from_coo(
            coo.row, coo.col, coo.data.astype(np.float64)
        ).scipy()
        for r in range(csr.shape[0]):
            for ind in range(csr.indptr[r], csr.indptr[r + 1]):
                self.assertEqual(data[r, csr.indices[ind]], csr.data[ind])

        self.assertEqual(len(coo.data), csr.nnz)

        data = {}

        coo = hazy.coo_from_file(dir_path + "/data/sample_data_2.bin").scipy()
        for i, j, v in zip(coo.row, coo.col, coo.data):
            data[(i, j)] = v

        print("Transforming from COO to CSR...")
        csr = hazytensor.DoubleCSR.from_coo(
            coo.row, coo.col, coo.data.astype(np.float64)
        ).scipy()
        for r in range(csr.shape[0]):
            for ind in range(csr.indptr[r], csr.indptr[r + 1]):
                self.assertEqual(data[r, csr.indices[ind]], csr.data[ind])

        self.assertEqual(len(coo.data), csr.nnz)


if __name__ == "__main__":
    unittest.main()
