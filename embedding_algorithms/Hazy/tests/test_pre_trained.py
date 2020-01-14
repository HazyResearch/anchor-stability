import csv
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
    def load_vocab(self, vocab_file):
        vocab = []
        fin = open(vocab_file, "r")
        for line in fin:
            w, _ = line.strip().split()
            vocab.append(w)
        fin.close()
        return vocab

    def save_to_file(self, emb, words, outfile):
        with open(outfile, "w") as f:
            writer = csv.writer(f, delimiter=" ")
            for i, word in enumerate(words):
                row = [word]
                row.extend(emb[i])
                writer.writerow(row)

    def test_pre_trained_different_corpus(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        timestamp = str(time.time())
        emb_filename1 = dir_path + "/" + "tmp_embeddings_1.txt_" + timestamp

        n_dimensions = 10
        n_words_orig = 5
        n_epochs = 0
        tol = 1e-10
        save_epochs = 0
        log_epochs = 200
        solver = "sgd"
        seed = 1234
        svrg_freq = 10
        batch_size = 10
        lr = 1
        beta = 0
        lam = 0
        lr_decay = 1
        n_threads = 4

        vocab_new = dir_path + "/data/sample_data_vocab.txt"
        words_new = self.load_vocab(vocab_new)

        # generate small embedding for initialization and write
        words_orig = ["william", "unusual", "tropical", "understand", "zayed"]
        np.random.seed(1234)
        emb_orig = np.random.rand(len(words_orig), n_dimensions)
        self.save_to_file(emb=emb_orig, words=words_orig, outfile=emb_filename1)

        # run for zero epochs
        # Generate synthetic data
        row, col, data = [], [], []
        for i in range(len(words_new)):
            for j in range(len(words_new)):
                row.append(i)
                col.append(j)
                data.append(i * j)
        row = np.array(row)
        col = np.array(col)
        data = np.array(data)
        coo = coo_matrix((data, (row, col)))
        csr = hazytensor.DoubleCSR.from_coo(
            coo.row, coo.col, coo.data.astype(np.float64)
        )
        coo_ = hazytensor.DoubleCOO.from_csr(csr)

        output_filename = "tmp_embeddings_2.txt_" + timestamp
        output = dir_path + "/" + output_filename
        new_corpus = True
        hazytensor.solve(
            csr,
            coo_,
            n_epochs,
            n_dimensions,
            tol,
            save_epochs,
            log_epochs,
            output,
            vocab_new,
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
            new_corpus,
        )

        # compare to un-aligned embedding (final_orig)
        emb_filename2 = glob.glob(dir_path + "/" + output_filename + "*final_orig")[0]
        m = load_emb(emb_filename2)
        # check that original (not aligned) embedding matches up with
        # original embedding in correct rows
        for i, w in enumerate(words_new):
            if w in words_orig:
                for d in range(n_dimensions):
                    orig_idx = words_orig.index(w)
                    self.assertAlmostEqual(m[i][d], emb_orig[orig_idx][d], places=5)

        clean_files(dir_path + "/" + output_filename + "*")
        clean_files(emb_filename1)

    def test_pi_pre_trained(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        n_words = 100
        timestamp = str(time.time())
        output_filename1 = "tmp_pi_embeddings_1.txt_" + timestamp

        n_epochs = 1000
        n_dimensions = n_words
        tol = 1e-10
        save_epochs = 0
        log_epochs = 200
        output1 = dir_path + "/" + output_filename1
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

        output_filename2 = "tmp_pi_embeddings_2.txt_" + timestamp

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
        m_ = np.matmul(np.matmul(m, scipy.diag(eigv)), m.T)
        error = np.linalg.norm(m_ - mat, "fro")
        self.assertLess(error, 0.1)
        clean_files(dir_path + "/" + output_filename1 + "*")
        clean_files(dir_path + "/" + output_filename2 + "*")


if __name__ == "__main__":
    unittest.main()
