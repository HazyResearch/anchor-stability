import glob
import os
import os.path
import subprocess
import unittest

import hazy
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

import hazytensor

from .utils import clean_files

# End to end test on text8


class MainTest(unittest.TestCase):
    def download_text8_data(self):
        if not os.path.isfile("text8"):
            os.system("wget http://mattmahoney.net/dc/text8.zip")
            os.system("unzip text8.zip")
            os.system("rm text8.zip")

    def test_text8(self):
        count_vocab = True
        cooccur = True
        shuffle = True

        dir_path = os.path.dirname(os.path.realpath(__file__))

        self.download_text8_data()
        # Method copied from GloVe
        if count_vocab:
            hazy.vocab_count(
                corpus_file="text8",
                vocab_file=dir_path + "/" + "vocab.txt",
                min_count=5,
            )

        # Method copied from GloVe
        if cooccur:
            hazy.cooccur(
                corpus_file="text8",
                cooccur_file=dir_path + "/" + "cooccur.bin",
                vocab_file=dir_path + "/" + "vocab.txt",
                memory=4.0,
                window_size=15,
            )

        # Method copied from GloVe
        if shuffle:
            hazy.shuffle(
                cooccur_in=dir_path + "/" + "cooccur.bin",
                cooccur_out=dir_path + "/" + "cooccur_shuffle.bin",
                memory=4.0,
            )

        # Loads the output cooccur from GloVe into a scipy COO.
        sm = hazy.coo_from_file(dir_path + "/" + "cooccur_shuffle.bin").scipy()

        hazy_csr = hazytensor.DoubleCSR.from_coo(
            sm.row, sm.col, sm.data.astype(np.float64)
        ).scipy()

        hazy.solve(
            cooccur=sm,
            n_epochs=10,
            n_dimensions=50,
            tol=1e-4,
            save_epochs=20,
            log_epochs=1,
            output=dir_path + "/" + "embeddings.txt",
            vocab=dir_path + "/" + "vocab.txt",
            solver="pi",
            seed=1234,
        )

        file = glob.glob(dir_path + "/embeddings.txt*.final")[0]
        process = subprocess.Popen(
            [
                "python",
                dir_path + "/../eval/intrinsic/ws_eval.py",
                "GLOVE",
                file,
                dir_path + "/../eval/intrinsic/testsets/ws/ws353_similarity.txt",
            ],
            stdout=subprocess.PIPE,
        )
        stdout = process.communicate()[0]

        self.assertGreater(float(stdout.strip().split()[-1]), 0.6)
        clean_files(dir_path + "/" + "vocab.txt")
        clean_files(dir_path + "/" + "cooccur.bin")
        clean_files(dir_path + "/" + "cooccur_shuffle.bin")
        clean_files(dir_path + "/" + "embeddings.txt" + "*")
