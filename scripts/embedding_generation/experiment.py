# basic framework for launching experiments across embedding algorithms
import argparse
import os

from anchor.algorithms import GloVe, MatrixCompletion, PowerIteration, Word2Vec
from anchor.embedding import DualEmbedding, Embedding
from anchor.system import Anchor, Ensemble, Initializer


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--w2v_cbow", action="store_true")
    parser.add_argument("--pi", action="store_true")
    parser.add_argument("--glove", action="store_true")
    parser.add_argument("--mc", action="store_true")
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--coo", type=str, required=True)
    parser.add_argument("--vocab", type=str, required=True)
    parser.add_argument("--warmstart", action="store_true")
    parser.add_argument("--base_emb", type=str)
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--resultdir", type=str, required=True)
    parser.add_argument("--homedir", type=str, required=True)
    parser.add_argument("--log_file", type=str)
    parser.add_argument("--ckpt", type=int, default=100)
    return parser.parse_args()


args = parse_args()

homedir = args.homedir
resultdir = args.resultdir

os.makedirs(resultdir, exist_ok=True)
if args.warmstart:
    ems = Anchor(
        emb_dir=resultdir,
        initializer=Initializer.WARMSTART,
        previous_embs=[Embedding(args.base_emb)],
    )

else:
    ems = Anchor(emb_dir=resultdir)

if args.pi:
    algo = PowerIteration(
        exec_dir=f"{homedir}/embedding_algorithms/Hazy/build/bin",
        tag=args.tag,
        epoch=args.epoch,
        dim=args.dim,
        seed=args.seed,
        threads=args.threads,
        checkpoint_interval=args.ckpt
    )

elif args.mc:
    # hyperparam sweep
    if args.lr:
        algo = MatrixCompletion(
            exec_dir=f"{homedir}/embedding_algorithms/Hazy/build/bin",
            tag=args.tag,
            epoch=args.epoch,
            dim=args.dim,
            seed=args.seed,
            threads=args.threads,
            lr=args.lr,
            checkpoint_interval=args.ckpt
        )
    # use defaults
    else:
        algo = MatrixCompletion(
            exec_dir=f"{homedir}/embedding_algorithms/Hazy/build/bin",
            tag=args.tag,
            epoch=args.epoch,
            dim=args.dim,
            seed=args.seed,
            threads=args.threads,
            checkpoint_interval=args.ckpt
        )

elif args.w2v_cbow:
    # hyperparam sweep
    if args.lr:
        algo = Word2Vec(
            exec_dir=f"{homedir}/embedding_algorithms/word2vec/word2vec",
            tag=args.tag,
            epoch=args.epoch,
            dim=args.dim,
            seed=args.seed,
            threads=args.threads,
            alpha=args.lr,
            cbow=True,
            checkpoint_interval=args.ckpt
        )
    else:  # use defaults
        algo = Word2Vec(
            exec_dir=f"{homedir}/embedding_algorithms/word2vec/word2vec",
            tag=args.tag,
            epoch=args.epoch,
            dim=args.dim,
            seed=args.seed,
            threads=args.threads,
            cbow=True,
            checkpoint_interval=args.ckpt
        )

elif args.glove:
    # hyperparam sweep
    if args.lr:
        algo = GloVe(
            exec_dir=f"{homedir}/embedding_algorithms/GloVe/build",
            tag=args.tag,
            epoch=args.epoch,
            dim=args.dim,
            seed=args.seed,
            threads=args.threads,
            eta=args.lr,
            checkpoint_interval=args.ckpt
        )
    # use defaults
    else:
        algo = GloVe(
            exec_dir=f"{homedir}/embedding_algorithms/GloVe/build",
            tag=args.tag,
            epoch=args.epoch,
            dim=args.dim,
            seed=args.seed,
            threads=args.threads,
            checkpoint_interval=args.ckpt
        )

ems.gen_embedding(
    data=args.text, algo=algo, coo=args.coo, vocab=args.vocab, log_file=args.log_file
)
