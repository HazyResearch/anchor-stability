import glob
import hashlib
import os
import time
import unittest

import hazy
import numpy as np
import scipy
from scipy import linalg, random, sparse
from scipy.sparse import coo_matrix

import hazytensor

from .utils import clean_files, load_emb

# Test power iteration


class MainTest(unittest.TestCase):
    def test_pi_synthetic(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        n_words = 100
        timestamp = str(time.time())
        output_filename = "tmp_pi_embeddings.txt_" + timestamp

        n_epochs = 2000
        n_dimensions = n_words
        tol = 1e-10
        save_epochs = 0
        log_epochs = 200
        output = dir_path + "/" + output_filename
        vocab = dir_path + "/data/sample_data_vocab.txt"
        solver = "pi"
        seed = 1234
        svrg_freq = 10
        batch_size = 10
        lr = 1
        beta = 0
        lam = 0
        lr_decay = 1
        n_threads = 4

        # Generate synthetic data
        x, _ = linalg.qr(np.random.randn(n_words, n_words))
        eigv = np.array(range(1, n_words + 1)[::-1]) * 10
        mat = np.matmul(np.matmul(x, scipy.diag(eigv)), x.T)
        u, s, vh = np.linalg.svd(mat, full_matrices=False)

        row, col, data = [], [], []
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                row.append(i)
                col.append(j)
                data.append(mat[i, j])
        row = np.array(row)
        col = np.array(col)
        data = np.array(data)
        coo = coo_matrix((data, (row, col)))

        csr = hazytensor.DoubleCSR.from_coo(
            coo.row, coo.col, coo.data.astype(np.float64)
        )
        coo_ = hazytensor.DoubleCOO.from_csr(csr)

        hazytensor.solve(
            csr,
            coo_,
            n_epochs,
            n_dimensions,
            tol,
            save_epochs,
            log_epochs,
            output,
            vocab,
            solver,
            seed,
            svrg_freq,
            batch_size,
            lr,
            beta,
            lam,
            lr_decay,
            n_threads,
        )

        emb_filename = glob.glob(dir_path + "/" + output_filename + "*final")[0]
        m = load_emb(emb_filename)
        error = 0.0
        m_ = np.matmul(np.matmul(m, scipy.diag(eigv)), m.T)
        error = np.linalg.norm(m_ - mat, "fro")
        self.assertLess(error, 0.1)
        clean_files(dir_path + "/" + output_filename + "*")

    def test_dpi_synthetic(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        n_words = 100
        timestamp = str(time.time())
        output_filename = "tmp_dpi_embeddings.txt_" + timestamp

        n_epochs = 2000
        n_dimensions = n_words
        tol = 1e-10
        save_epochs = 0
        log_epochs = 200
        output = dir_path + "/" + output_filename
        vocab = dir_path + "/data/sample_data_vocab.txt"
        solver = "dpi"
        seed = 1234
        svrg_freq = 10
        batch_size = 10
        lr = 1
        beta = 0
        lam = 0
        lr_decay = 1
        n_threads = 4

        # Generate synthetic data
        x, _ = linalg.qr(np.random.randn(n_words, n_words))
        eigv = np.array(range(1, n_words + 1)[::-1]) * 10
        mat = np.matmul(np.matmul(x, scipy.diag(eigv)), x.T)
        u, s, vh = np.linalg.svd(mat, full_matrices=False)

        row, col, data = [], [], []
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                row.append(i)
                col.append(j)
                data.append(mat[i, j])
        row = np.array(row)
        col = np.array(col)
        data = np.array(data)
        coo = coo_matrix((data, (row, col)))

        csr = hazytensor.DoubleCSR.from_coo(
            coo.row, coo.col, coo.data.astype(np.float64)
        )
        coo_ = hazytensor.DoubleCOO.from_csr(csr)

        hazytensor.solve(
            csr,
            coo_,
            n_epochs,
            n_dimensions,
            tol,
            save_epochs,
            log_epochs,
            output,
            vocab,
            solver,
            seed,
            svrg_freq,
            batch_size,
            lr,
            beta,
            lam,
            lr_decay,
            n_threads,
        )

        emb_filename = glob.glob(dir_path + "/" + output_filename + "*final")[0]
        m = load_emb(emb_filename)
        error = 0.0
        m_ = np.matmul(np.matmul(m, scipy.diag(eigv)), m.T)
        error = np.linalg.norm(m_ - mat, "fro")
        self.assertLess(error, 0.1)
        clean_files(dir_path + "/" + output_filename + "*")


if __name__ == "__main__":
    unittest.main()
