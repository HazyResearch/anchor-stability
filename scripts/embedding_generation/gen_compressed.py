"""
Run the embedding compression using smallfry's implementation.
"""

from smallfry.compress import compress_uniform
from smallfry.utils import load_embeddings, save_embeddings

import argparse
import io
import numpy as np
import os

import sys
from anchor.embedding import Embedding

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_path', type=str, required=True, help='Embedding to compress')
    parser.add_argument('--base_emb_path', type=str, help='Base embedding to use for alignment and compression intervals')
    parser.add_argument('--bitrate', type=int, default=1, required=True, help='Precision of embedding')
    parser.add_argument('--out', type=str, required=True, help='Where to save compressed embedding')
    parser.add_argument('--seed', type=int, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    emb2 = Embedding(args.emb_path)

    if args.base_emb_path is not None:
        emb1 = Embedding(args.base_emb_path)

    # align embeddings first
    if args.base_emb_path is not None:
        print('Aligning!')
        emb2.align(emb1)

    if args.base_emb_path is not None:
        Xq, frob_squared_error, elapsed = compress_uniform(X=emb2.m, bit_rate=args.bitrate,
                adaptive_range=True, X_0=emb1.m)
    else:
        Xq, frob_squared_error, elapsed = compress_uniform(X=emb2.m, bit_rate=args.bitrate,
            adaptive_range=True)

    print(frob_squared_error, elapsed)
    print(Xq.shape)

    # save compressed embedding
    save_embeddings(path=args.out, embeds=Xq, wordlist=emb2.iw)

if __name__ == '__main__':
    main()
