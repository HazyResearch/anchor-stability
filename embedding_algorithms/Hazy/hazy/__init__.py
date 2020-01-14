import hazytensor
import numpy as np


def solve(
    cooccur,
    n_epochs=50,
    n_dimensions=300,
    tol=1e-4,
    save_epochs=0,
    log_epochs=1,
    output="embeddings.txt",
    vocab="vocab.txt",
    solver="pi",
    seed=1234,
    svrg_freq=0,
    batch_size=128,
    lr=0.1,
    beta=0,
    lam=0,
    lr_decay=1,
    n_threads=4,
):
    print("Transforming from COO to CSR...")
    csr = hazytensor.DoubleCSR.from_coo(
        cooccur.row, cooccur.col, cooccur.data.astype(np.float64)
    )
    print("Running ppmi...")
    csr.ppmi()
    print("Creating COO...")
    coo = hazytensor.DoubleCOO.from_csr(csr)
    print("Shuffling COO...")
    coo.shuffle_inplace(seed)
    print("Running solver..")
    return hazytensor.solve(
        csr,
        coo,
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


def coo_from_file(*args, **kwargs):
    return hazytensor.DoubleCOO.from_file(*args, **kwargs)


def cooccur(*args, **kwargs):
    return hazytensor.cooccur(*args, **kwargs)


def vocab_count(*args, **kwargs):
    return hazytensor.vocab_count(*args, **kwargs)


def shuffle(*args, **kwargs):
    return hazytensor.shuffle(*args, **kwargs)


def orthogonal_procrustes(*args, **kwargs):
    return hazytensor.orthogonal_procrustes(*args, **kwargs)


def procrustes_pi(*args, **kwargs):
    return hazytensor.procrustes_pi(*args, **kwargs)


def DoubleDenseMatrix(*args, **kwargs):
    return hazytensor.DoubleDenseMatrix(*args, **kwargs)
