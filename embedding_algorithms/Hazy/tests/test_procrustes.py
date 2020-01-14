import glob
import os
import time
import unittest

import hazy
import numpy as np
import scipy
import scipy.linalg as la
from scipy import random, sparse
from scipy.sparse import coo_matrix

import hazytensor

from .utils import clean_files, load_emb

# Test procrustes methods


class MainTest(unittest.TestCase):
    def compare_embs(self, procrustes_embs, original_embs):
        proc_diffs = []
        orig_diffs = []
        for i in range(len(procrustes_embs)):
            for j in range(i + 1, len(procrustes_embs)):
                proc_diff = np.linalg.norm(
                    procrustes_embs[i] - procrustes_embs[j], "fro"
                )
                orig_diff = np.linalg.norm(original_embs[i] - original_embs[j], "fro")
                proc_diffs.append(proc_diff)
                orig_diffs.append(orig_diff)
                self.assertLessEqual(proc_diff, orig_diff)

        # Check max difference -- PI very stable
        print(max(proc_diffs), max(orig_diffs))

    def test_orthogonal_procrustes(self):
        np.random.seed(1234)
        m = 100
        n = 10
        X = np.random.randn(m, n)
        Xis = []
        Xi_tildes = []
        # Generate random rotation matrices
        for i in range(10):
            omega, _ = la.qr(np.random.rand(n, n))
            Xi = np.dot(X, omega)
            Xi = np.array(Xi).astype(np.float64)
            Xis.append(Xi)
            Xi_tilde = hazy.orthogonal_procrustes(hazy.DoubleDenseMatrix(Xi)).numpy()
            Xi_tildes.append(Xi_tilde)

        error_orig = []
        error_proc = []  # Error procrustes
        for i in range(len(Xi_tildes)):
            for j in range(i, len(Xi_tildes)):
                error_proc.append(la.norm(Xi_tildes[i] - Xi_tildes[j], ord="fro"))
                error_orig.append(la.norm(Xis[i] - Xis[j], ord="fro"))
        # We expect matrices output from Procrustes method to be closer
        # to each other (by frobenius norm)
        for i in range(len(error_proc)):
            self.assertLessEqual(error_proc[i], error_orig[i])
        self.assertLess(max(error_proc), max(error_orig))

    def test_procrustes_pi(self):
        np.random.seed(1234)
        m = 100
        n = 10
        X = np.random.randn(m, n)
        Xis = []
        Xi_tildes = []
        # Generate random rotation matrices
        # For PI these are diagonal matrices with 1 and -1s on the diagonal
        for i in range(10):
            diag_ = np.random.choice([-1, 1], size=n)
            omega = np.diag(diag_)
            Xi = np.dot(X, omega)
            Xi = np.array(Xi).astype(np.float64)
            Xis.append(Xi)
            Xi_tilde = hazy.procrustes_pi(hazy.DoubleDenseMatrix(Xi)).numpy()
            Xi_tildes.append(Xi_tilde)

        error_orig = []
        error_proc = []  # Error procrustes
        for i in range(len(Xi_tildes)):
            for j in range(i, len(Xi_tildes)):
                error_proc.append(la.norm(Xi_tildes[i] - Xi_tildes[j], ord="fro"))
                error_orig.append(la.norm(Xis[i] - Xis[j], ord="fro"))
        # We expect matrices output from Procrustes method to be closer
        # to each other (by frobenius norm)
        for i in range(len(error_proc)):
            self.assertLessEqual(error_proc[i], error_orig[i])
        self.assertLess(max(error_proc), max(error_orig))

    def test_orthogonal_procrustes_integration(self):
        np.random.seed(1234)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        n_words = 10
        timestamp = str(time.time())
        output_1_filename = "tmp_sgd_embeddings_1.txt_" + timestamp
        output_2_filename = "tmp_sgd_embeddings_2.txt_" + timestamp

        n_epochs = 10000
        n_dimensions = n_words
        tol = 1e-20
        save_epochs = 0
        log_epochs = 0
        output_1 = dir_path + "/" + output_1_filename
        output_2 = dir_path + "/" + output_2_filename

        vocab = dir_path + "/data/sample_data_vocab.txt"
        solver = "sgd"
        seed = 1234
        svrg_freq = 0
        batch_size = 1
        lr = 1e-1
        beta = 0
        lam = 0
        lr_decay = 1000

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

        proc_embs = []
        orig_embs = []
        output_files = []

        # Change number of threads (Hogwild should cause results to be slightly different)
        for n_threads in range(1, 50, 10):
            output_filename = "tmp_sgd_embeddings_{}.txt_{}".format(
                n_threads, timestamp
            )
            output_files.append(output_filename)
            output = dir_path + "/" + output_filename
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
            emb_filename_proc = glob.glob(dir_path + "/" + output_filename + "*final")[
                0
            ]
            emb_filename_orig = glob.glob(
                dir_path + "/" + output_filename + "*final_orig"
            )[0]
            m_proc = load_emb(emb_filename_proc)
            m_orig = load_emb(emb_filename_orig)
            proc_embs.append(m_proc)
            orig_embs.append(m_orig)
        self.compare_embs(proc_embs, orig_embs)

        n_threads = 1
        # Change seed (affects embedding initialization)
        seeds = [1234, 4321, 777, 1111]
        proc_embs = []
        orig_embs = []
        for seed in seeds:
            output_filename = "tmp_sgd_embeddings_{}.txt_{}".format(seed, timestamp)
            output_files.append(output_filename)
            output = dir_path + "/" + output_filename
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
            emb_filename_proc = glob.glob(dir_path + "/" + output_filename + "*final")[
                0
            ]
            emb_filename_orig = glob.glob(
                dir_path + "/" + output_filename + "*final_orig"
            )[0]
            m_proc = load_emb(emb_filename_proc)
            m_orig = load_emb(emb_filename_orig)
            proc_embs.append(m_proc)
            orig_embs.append(m_orig)
        self.compare_embs(proc_embs, orig_embs)

        # Change learning rate
        n_threads = 1
        lrs = [0.1, 0.05, 0.01]
        log_epochs = 1000
        n_epochs = 10000
        lr_decay = 1000
        proc_embs = []
        orig_embs = []
        for lr in lrs:
            output_filename = "tmp_sgd_embeddings_{}.txt_{}".format(lr, timestamp)
            output_files.append(output_filename)
            output = dir_path + "/" + output_filename
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
            emb_filename_proc = glob.glob(dir_path + "/" + output_filename + "*final")[
                0
            ]
            emb_filename_orig = glob.glob(
                dir_path + "/" + output_filename + "*final_orig"
            )[0]
            m_proc = load_emb(emb_filename_proc)
            m_orig = load_emb(emb_filename_orig)
            proc_embs.append(m_proc)
            orig_embs.append(m_orig)
        self.compare_embs(proc_embs, orig_embs)

        for filename in output_files:
            clean_files(dir_path + "/" + filename + "*")

    def test_procrustes_pi_integration(self):
        np.random.seed(1234)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        n_words = 10
        timestamp = str(time.time())

        n_epochs = 10000
        n_dimensions = n_words
        tol = 1e-20
        save_epochs = 0
        log_epochs = 0

        vocab = dir_path + "/data/sample_data_vocab.txt"
        solver = "pi"
        seed = 1234
        svrg_freq = 10
        batch_size = 10
        lr = 1
        beta = 0
        lam = 0
        lr_decay = 1
        n_threads = 1

        # Generate synthetic data
        x, _ = la.qr(np.random.randn(n_words, n_words))
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

        # Only source of nondeterminism for PI is the initial embedding
        seeds = [1234, 4321, 777, 1111]

        output_files = []
        proc_embs = []
        orig_embs = []
        for seed in seeds:
            output_filename = "tmp_pi_embeddings_{}.txt_{}".format(seed, timestamp)
            output_files.append(output_filename)
            output = dir_path + "/" + output_filename
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
            emb_filename_proc = glob.glob(dir_path + "/" + output_filename + "*final")[
                0
            ]
            emb_filename_orig = glob.glob(
                dir_path + "/" + output_filename + "*final_orig"
            )[0]
            m_proc = load_emb(emb_filename_proc)
            m_orig = load_emb(emb_filename_orig)
            proc_embs.append(m_proc)
            orig_embs.append(m_orig)
        self.compare_embs(proc_embs, orig_embs)

        for filename in output_files:
            clean_files(dir_path + "/" + filename + "*")


if __name__ == "__main__":
    unittest.main()
