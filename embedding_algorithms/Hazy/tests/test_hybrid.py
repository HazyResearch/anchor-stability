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
    def test_hybrid_real(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        n_words = 524
        timestamp = str(time.time())
        output_filename = "tmp_hyper_embeddings.txt_" + timestamp

        n_epochs = 2000
        n_dimensions = 50
        tol = 1e-3
        save_epochs = 0
        log_epochs = 200
        output = dir_path + "/" + output_filename
        vocab = dir_path + "/data/sample_data_vocab.txt"
        solver = "sgd"
        seed = 1234
        svrg_freq = 1
        batch_size = 10
        lr = 1e-2
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

        emb_filename = sorted(glob.glob(dir_path + "/" + output_filename + "*final"))[
            -1
        ]
        print("FILENAME: " + emb_filename)
        m = load_emb(emb_filename)
        error = 0.0
        for i, j, v in zip(coo.row, coo.col, coo.data):
            error += (np.dot(m[i], m[j]) - v) ** 2
        error = np.sqrt(error)
        print(error)
        self.assertLess(error, 290)

        clean_files(dir_path + "/" + output_filename + "*")


if __name__ == "__main__":
    unittest.main()
