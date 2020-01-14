import argparse
import functools
import glob
import math
import multiprocessing as mp
import operator
import os
import pickle
import subprocess

import numpy as np
from utils import run_task

parser = argparse.ArgumentParser()
parser.add_argument(
    "-t", "--task", type=str, default="ws353.txt", help="Task. Default: ws353.txt"
)
parser.add_argument("--algo", type=str, help="Algo")
args = parser.parse_args()

resultdir = "/dfs/scratch1/mleszczy/sigmod/embs/dim_sweep_seed"
taskdir = "/dfs/scratch1/mleszczy/anchor/embedding_algorithms/Hazy/eval/intrinsic"
taskfile = os.path.join(taskdir, "testsets/ws", args.task)

def check_seeds(final_results):
    num_seeds = None
    for lr, vals in final_results.items():
        if num_seeds is None:
            num_seeds = vals[2]
        else:
            assert (
                vals[2] == num_seeds
            ), "Number of seeds is not the same for all lr values"

def get_corr(f):
    terms = f.split("_")
    if args.algo == "w2v_cbow":
        lr = float(terms[terms.index("lr") + 1].split(".100.w.txt")[0])
    else:
        lr = float(terms[terms.index("lr") + 1].split(".100.txt")[0])
    seed = int(terms[terms.index("seed") + 1])
    correlation = run_task(taskdir, taskfile, f)
    return (lr, seed, correlation)

dims = [25, 50, 100, 200, 500, 1000, 2000]

pool = mp.Pool(processes=4)
total_results = []
for dim in dims:
    scores = {}
    if args.algo == "w2v_cbow":
        results = glob.glob(f"{resultdir}/{args.algo}*dim_{dim}_*100.w.txt")
    else:
        results = glob.glob(f"{resultdir}/{args.algo}*dim_{dim}_*100.txt")
    corr_runs = [pool.apply_async(get_corr, args=(f,)) for f in results]
    corr_results = [p.get() for p in corr_runs]
    for lr, _, val in corr_results:
        if lr in scores:
            scores[lr].append(val)
        else:
            scores[lr] = [val]
    final_results = {
        lr: (np.mean(vals), np.std(vals), len(vals)) for lr, vals in scores.items()
    }
    check_seeds(final_results)
    sorted_x = sorted(
        final_results.items(),
        key=lambda x: x[1][0] if not math.isnan(x[1][0]) else 0,
        reverse=True,
    )
    print((dim,
            sorted_x[0][0],
            sorted_x[0][1][0],
            sorted_x[0][1][2],
            [sorted_x[i][0] for i in range(len(sorted_x))]))
    total_results.append(
        (
            dim,
            sorted_x[0][0],
            sorted_x[0][1][0],
            sorted_x[0][1][2],
            [sorted_x[i][0] for i in range(len(sorted_x))],
        )
    )

print(total_results)
result_path = (
    f"{resultdir}/{args.algo}_{args.task}_results.pkl"
)
pickle.dump(total_results, open(result_path, "wb"))
