import os
import os.path

import hazy
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

import hazytensor


def download_text8_data():
    if not os.path.isfile("text8"):
        os.system("wget http://mattmahoney.net/dc/text8.zip")
        os.system("unzip text8.zip")
        os.system("rm text8.zip")


count_vocab = True
cooccur = True
shuffle = True

download_text8_data()
# Method copied from GloVe
if count_vocab:
    hazy.vocab_count(corpus_file="text8", vocab_file="vocab.txt", min_count=5)

# Method copied from GloVe
if cooccur:
    hazy.cooccur(
        corpus_file="text8",
        cooccur_file="cooccur.bin",
        vocab_file="vocab.txt",
        memory=4.0,
        window_size=15,
    )

# Method copied from GloVe
if shuffle:
    hazy.shuffle(
        cooccur_in="cooccur.bin", cooccur_out="cooccur_shuffle.bin", memory=4.0
    )

# Loads the output cooccur from GloVe into a scipy COO.
sm = hazy.coo_from_file("cooccur_shuffle.bin").scipy()

hazy_csr = hazytensor.DoubleCSR.from_coo(
    sm.row, sm.col, sm.data.astype(np.float64)
).scipy()

hazy.solve(
    cooccur=sm,
    n_epochs=10,
    n_dimensions=300,
    tol=1e-4,
    output="embeddings.txt",
    vocab="vocab.txt",
    solver="pi",
    seed=1234,
)
