# this generates shell commands to be run either with xargs or qsub
# to train embeddings for dimension experiment

import os
import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--year", type=int, choices=[2017, 2018], required=True)
    parser.add_argument("--dim", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--algo", type=str)
    return parser.parse_args()

args = parse_args()

epoch = 50
ckpt = 10

resultdir = f"/dfs/scratch1/mleszczy/sigmod/embs/wiki/wiki_{args.year}"
homedir = "/dfs/scratch1/mleszczy"
os.makedirs(resultdir, exist_ok=True)

threads = 56

if args.seed:
   seeds = [args.seed]
else:
   seeds = range(1234, 1237)

if args.dim:
   dims = [args.dim]
else:
   # dims = [50, 200, 400, 800]
   dims = [1200, 1600]

lrs = {"mc": [0.2], 'w2v_cbow': [0.05], 'glove': [0.01]}
#algos = ['mc', 'w2v_cbow']
algos = [args.algo]
if args.year == 2017:
    coo = f"{homedir}/data/wiki_2017/cooccurrence_minCount_5_ws_15.bin_400k"
    vocab = f"{homedir}/data/wiki_2017/vocab_minCount_5_ws_15.txt_400k"
    text = f"{homedir}/data/wiki_2017/wiki.en.txt"
else:
    coo = f"{homedir}/data/wiki_2018/20190128/wiki.en.txt_cooccurrence_minCount_5_ws_15.bin_400k"
    vocab = f"{homedir}/data/wiki_2018/20190128/wiki.en.txt_vocab_minCount_5_ws_15.txt_400k"
    text = f"{homedir}/data/wiki_2018/20190128/wiki.en.txt"

for algo in algos:
    for seed in seeds:
        for dim in dims:
            if algo == 'pi':
               tag = f"{os.path.basename(text)}_{args.year}_seed_{seed}_dim_{dim}"
               log_file = f"{resultdir}/anchor_{algo}_{tag}.log"
               print(
                     f"python {homedir}/anchor/scripts/experiment.py --ckpt {ckpt} --dim {dim} --{algo} --epoch {epoch} --text {text} --coo {coo} --vocab {vocab} --tag {algo}_{tag} --resultdir {resultdir} --homedir {homedir}/anchor --log_file {log_file} --seed {seed} --threads {threads}"
                  )
            else:
               for lr in lrs[algo]:
                  tag = f"{os.path.basename(text)}_{args.year}_seed_{seed}_dim_{dim}_lr_{lr}"
                  log_file = f"{resultdir}/anchor_{algo}_{tag}.log"
                  print(
                     f"python {homedir}/anchor/scripts/experiment.py --ckpt {ckpt} --dim {dim} --{algo} --epoch {epoch} --text {text} --coo {coo} --vocab {vocab} --tag {algo}_{tag} --resultdir {resultdir} --homedir {homedir}/anchor --log_file {log_file} --lr {lr} --seed {seed} --threads {threads}"
                  )
