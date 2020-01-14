import glob
import hashlib
import os
import time
import unittest

import hazy
import numpy as np
import scipy
from scipy import sparse
from scipy.sparse import coo_matrix

import hazytensor

from .utils import clean_files, load_emb

# Test sgd


class MainTest(unittest.TestCase):
    def test_sgd_synthetic(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        n_words = 10
        timestamp = str(time.time())
        output_filename = "tmp_sgd_embeddings.txt_" + timestamp

        n_epochs = 2000
        n_dimensions = n_words
        tol = 1e-10
        save_epochs = 0
        log_epochs = 200
        output = dir_path + "/" + output_filename
        vocab = dir_path + "/data/sample_data_vocab.txt"
        solver = "sgd"
        seed = 1234
        svrg_freq = 0
        batch_size = 1
        lr = 1e-2
        beta = 0
        lam = 0
        lr_decay = 100
        n_threads = 4

        # Generate synthetic data
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
        for i, j, v in zip(coo.row, coo.col, coo.data):
            error += (np.dot(m[i], m[j]) - v) ** 2
        error = np.sqrt(error)
        self.assertLess(error, 0.2)
        clean_files(dir_path + "/" + output_filename + "*")

    def test_sgd_real(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        n_words = 524
        timestamp = str(time.time())
        output_filename1 = "tmp_sgd_embeddings_1.txt_" + timestamp

        n_epochs = 2000
        n_dimensions = 50
        tol = 1e-4
        save_epochs = 0
        log_epochs = 200
        output1 = dir_path + "/" + output_filename1
        vocab = dir_path + "/data/sample_data_vocab.txt"
        solver = "sgd"
        seed = 1234
        svrg_freq = 0
        batch_size = 10
        lr = 1e-3
        beta = 0
        lam = 0
        lr_decay = 100
        n_threads = 4

        # Loads the output cooccur from GloVe into a scipy COO.
        coo = hazy.coo_from_file(dir_path + "/data/sample_data_cooccur.bin").scipy()
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
            output1,
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

        emb_filename1 = glob.glob(dir_path + "/" + output_filename1 + "*final")[0]
        m = load_emb(emb_filename1)
        error = 0.0
        for i, j, v in zip(coo.row, coo.col, coo.data):
            error += (np.dot(m[i], m[j]) - v) ** 2
        error = np.sqrt(error)
        self.assertLess(error, 290)

        # Train starts with pre-trained model
        output_filename2 = "tmp_sgd_embeddings_2.txt_" + timestamp
        output2 = dir_path + "/" + output_filename2

        hazytensor.solve(
            csr,
            coo_,
            n_epochs,
            n_dimensions,
            tol,
            save_epochs,
            log_epochs,
            output2,
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
            emb_filename1,
        )

        emb_filename2 = glob.glob(dir_path + "/" + output_filename2 + "*final")[0]
        m = load_emb(emb_filename2)
        error = 0.0
        for i, j, v in zip(coo.row, coo.col, coo.data):
            error += (np.dot(m[i], m[j]) - v) ** 2
        error = np.sqrt(error)
        self.assertLess(error, 290)
        clean_files(dir_path + "/" + output_filename1 + "*")
        clean_files(dir_path + "/" + output_filename2 + "*")


if __name__ == "__main__":
    unittest.main()
