import argparse
import glob
import os
import pickle
import multiprocessing as mp

import numpy as np
import pandas as pd

from anchor.embedding import Embedding
from utils import run_task, check_sent_complete, check_ner_complete

# Main file to compute the embedding metrics and downstream metrics between pairs
# of embeddings and save the results to panda dataframes

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dist", type=str, required=True,
        choices=['eigen_overlap', 'sem_disp', 'pred', 'weighted_eigen_overlap',
        'quality', 'fro_norm', 'pip', 'spec_norm', 'knn', 'anchor_eigen_overlap'],
        help='Distance metric between embeddings or models')
    parser.add_argument("--resultdir1", type=str, help='Directory for first embedding or model')
    parser.add_argument("--resultdir2", type=str,help='Directory for second embedding or model')
    parser.add_argument("--out", type=str, default=".", help='Result directory for dataframe')
    parser.add_argument('-t', '--task', type=str,
        help='Task for intrinsic or extrinsic comparison')
    parser.add_argument('--algo', type=str, help='Emb. algorithm', required=True)
    parser.add_argument('--compressed', action='store_true')
    parser.add_argument("--compress_type", type=str, help='Compression type', default='uniform',
        choices=['kmeans', 'uniform'])
    parser.add_argument("--dim", type=int)
    parser.add_argument("--bitrate", type=int)
    parser.add_argument("--model", type=str, help='Type of model, e.g. CNN')
    parser.add_argument("--exp", type=float, help="Exponent for weighting eigenvalues")
    parser.add_argument("--nprocs", type=int, default=20, help="Number of processes")
    parser.add_argument("--wiki", type=bool, default=True, help='Use wiki embeddings')
    parser.add_argument("--no_norm", action="store_true", help="Do not normalize the overlap metric")
    parser.add_argument("--symmetric", action='store_true')
    parser.add_argument("--lr", type=float, help="Learning rate for downstream model")
    parser.add_argument("--no_align", action='store_true')
    parser.add_argument("--truncate", type=int, help='Truncate embeddings to this number of words', default=-1)
    parser.add_argument("--tag", type=str, help='Additional tag to add to distance metric')
    parser.add_argument("--random", action='store_true', help='Randomly sampling vocab')
    parser.add_argument("--nquery", type=int, help='Number of queries for the knn metric', default=1000)
    parser.add_argument("--nneighbors", type=int, help='Number of neighbors to compare for the knn metric', default=10)
    parser.add_argument("--validation", action='store_true', help='Use the validation predictions')
    parser.add_argument("--seed_test", action='store_true',
        help='Use different seed for 2018 embeddings to test seed effect')
    parser.add_argument("--nseeds", type=int)
    parser.add_argument("--same_norm", action='store_true', help='Use embeddings that have the same norm for algo comparison')
    parser.add_argument("--scale", type=float, help='Magnitude to normalize embeddings to.')
    return parser.parse_args()

def ner_stability(modelpath1, modelpath2, val=False):
    """Return the downstream prediction disagreement for the NER task."""
    if val:
        preds1 = f"{modelpath1}/dev.tsv"
        preds2 = f"{modelpath2}/dev.tsv"
    else:
        preds1 = f"{modelpath1}/test.tsv"
        preds2 = f"{modelpath2}/test.tsv"
    check_ner_complete(modelpath1, modelpath2)
    file1_lines = open(preds1, 'r').readlines()
    file2_lines = open(preds2, 'r').readlines()

    # compare predictions
    mismatch = 0
    total = 0
    for line1, line2 in zip(file1_lines, file2_lines):
        if len(line1.split(' ')) > 3:
            pred1 = line1.split(' ')[2]
            pred2 = line2.split(' ')[2]
            # skip over cases where true value is "O"
            if line1.split(' ')[1] != 'O':
                if pred1 != pred2:
                    mismatch += 1
                total +=  1
    dist = mismatch / float(total) * 100
    return dist

def get_dist_tag(args):
    """Return the tag to use for the 'distance' measure in the dataframe."""
    # downstream tasks
    if args.model is not None:
        # measure stability
        if args.dist == 'pred':
            dist_tag = f'{args.model}_{args.task}'
        # measure quality
        else:
            dist_tag = f'{args.model}_{args.task}_quality'
    else:
        dist_tag = args.dist
    # append additional tags
    if args.exp:
        dist_tag += f'_{args.exp}'
    if args.no_norm:
        dist_tag += f'_no_norm'
    if args.symmetric:
        dist_tag += f'_sym'
    if args.truncate > -1:
        dist_tag += f'_top_{args.truncate}'
    if args.random:
        dist_tag += '_rand'
    if args.dist == 'knn':
        dist_tag += f'_nquery_{args.nquery}_nn_{args.nneighbors}'
    if args.tag is not None:
        dist_tag += f'_{args.tag}'
    if args.validation:
        dist_tag += '_val'
    if args.seed_test:
        dist_tag += '_seed_test'
    return dist_tag

def get_final_dist_tag(dist_tag, args):
    """Add additional file identifier tags for saving the results. NOT used in dataframe."""
    if args.dim:
        dist_tag += f'_dim_{args.dim}'
    if args.compressed:
        dist_tag += f'_{args.compress_type}'
    if args.lr:
        dist_tag += f'_lr_{args.lr}'
    if args.wiki:
        dist_tag += '_wiki'
    if args.no_align:
        dist_tag += 'no_align'
    return dist_tag

def run_seed(algo, seed, dim, lr, bitrate=None, compress_type=None, args=None):
    # set seeds for model
    seed1 = seed
    if args.seed_test:
        seed2 = seed + 1000
    else:
        seed2 = seed

    # get paths for embeddings or models
    if algo != 'ft_sg':
        if algo == "w2v_cbow":
            end_tag = "50.w.txt"
        elif algo == "mc":
            end_tag = "50.txt"
        else:
            end_tag = "050.txt"

        path1 = f"{args.resultdir1}/{algo}_wiki.en.txt_2017_seed_{seed}_dim_{dim}_lr_{lr}.{end_tag}"
        path2 = f"{args.resultdir2}/{algo}_wiki.en.txt_2018_seed_{seed}_dim_{dim}_lr_{lr}.{end_tag}"
    else:
        path1 = f"{args.resultdir1}/ft_sg_wiki.en.txt_2017_seed_{seed}_dim_{dim}.vec"
        path2 = f"{args.resultdir2}/ft_sg_wiki.en.txt_2018_seed_{seed}_dim_{dim}.vec"
    # do embedding comparison
    if args.model is None:
        anchor_path = f'{args.resultdir1}/{args.algo}_anchor_seed_{seed}_top_{args.truncate}.pkl'
    if args.compressed:
        path1 += f'_br_{bitrate}'
        path2 += f'_br_{bitrate}_same_range'
        print(path2)
    else:
        path1 += f'_br_32'
        if not args.no_align:
            path2 += f'_br_32_same_range'
        print(path2)

    if args.same_norm:
        path1 += f'_norm_{args.scale}'
        path2 += f'_norm_{args.scale}'

    # get downstream model quality of second model
    elif args.dist == 'quality':
        # TODO(mleszczy): clean up this hideous code...
        assert args.model is not None and args.task is not None and args.lr is not None, "Must provide model, task, and lr for quality eval"
        if args.task != 'ner':
            try:
                modelpath1 = f"{path1}/{args.task}/model_{args.model}_dropout_0.5_seed_{seed1}_lr_{args.lr}"
                ff = open(f'{modelpath1}.log', 'r')
            except:
                modelpath1 = f"{path1}/{args.task}/model_{args.model}_dropout_0.5_seed_{seed1}_data_{seed1}_lr_{args.lr}"
                ff = open(f'{modelpath1}.log', 'r')
            dat = [_.strip() for _ in ff]
            quality1 = 1-float(dat[-2].strip().split(': ')[1])*100
            # TODO(mleszczy): should we average errors? currently only use error on latest embedding
            try:
                modelpath2 = f"{path2}/{args.task}/model_{args.model}_dropout_0.5_seed_{seed2}_lr_{args.lr}"
                ff = open(f'{modelpath2}.log', 'r')
            except:
                modelpath2 = f"{path2}/{args.task}/model_{args.model}_dropout_0.5_seed_{seed2}_data_{seed2}_lr_{args.lr}"
                ff = open(f'{modelpath2}.log', 'r')
            dat = [_.strip() for _ in ff]
            try:
                dist = (1-float(dat[-2].strip().split(': ')[1]))*100
            except:
                print(modelpath2)
                exit()
        else:
            modelpath2 = f"{path2}/ner_{args.model}_lr_{args.lr}/eval.log"
            ff = open(modelpath2, 'r')
            dat = [_.strip() for _ in ff]
            lr = float(os.path.basename(os.path.dirname(modelpath2)).split('_')[-1])
            assert 'f1-score' in dat[-7] and 'MICRO_AVG' in dat[-7], 'Parsing NER incorrect'
            dist = float(dat[-7].strip().split(' ')[-1])*100
            print(modelpath2, dist)

    # compute downstream stability
    elif args.model is not None:
            assert args.model is not None and args.task is not None and (args.lr is not None), "Must provide model, task, and lr for prediction eval"
            if args.task != 'ner':
                # load validation predictions
                if args.validation:
                    # load model trained on embedding
                    modelpath1 = f"{path1}/{args.task}/model_{args.model}_dropout_0.5_seed_{seed1}_lr_{args.lr}"
                    preds1 = np.array(pickle.load(open(f'{modelpath1}_eval.val.pred', "rb")))
                    modelpath2 = f"{path2}/{args.task}/model_{args.model}_dropout_0.5_seed_{seed2}_lr_{args.lr}"
                    preds2 = np.array(pickle.load(open(f'{modelpath2}_eval.val.pred', "rb")))
                    print(len(preds1), len(preds2))
                    dist = (1 - np.sum(preds1 == preds2) / float(len(preds1)))*100
                    # make sure logs are complete
                    assert check_sent_complete(modelpath1, modelpath2)
                else:
                    # load model trained on embedding
                    # hacky soln to deal with new naming w/ data seed
                    try:
                        modelpath1 = f"{path1}/{args.task}/model_{args.model}_dropout_0.5_seed_{seed1}_lr_{args.lr}"
                        preds1 = np.array(pickle.load(open(f'{modelpath1}.pred', "rb")))
                    except:
                        modelpath1 = f"{path1}/{args.task}/model_{args.model}_dropout_0.5_seed_{seed1}_data_{seed1}_lr_{args.lr}"
                        preds1 = np.array(pickle.load(open(f'{modelpath1}.pred', "rb")))
                    try:
                        modelpath2 = f"{path2}/{args.task}/model_{args.model}_dropout_0.5_seed_{seed2}_lr_{args.lr}"
                        preds2 = np.array(pickle.load(open(f'{modelpath2}.pred', "rb")))
                    except:
                        modelpath2 = f"{path2}/{args.task}/model_{args.model}_dropout_0.5_seed_{seed2}_data_{seed2}_lr_{args.lr}"
                        print(modelpath2)
                        preds2 = np.array(pickle.load(open(f'{modelpath2}.pred', "rb")))
                    dist = (1 - np.sum(preds1 == preds2) / float(len(preds1)))*100
                    assert check_sent_complete(modelpath1, modelpath2)
            else:
                modelpath1 = f"{path1}/ner_{args.model}_lr_{args.lr}"
                modelpath2 = f"{path2}/ner_{args.model}_lr_{args.lr}"
                dist = ner_stability(modelpath1, modelpath2, val=args.validation)

    # Compute embedding distance measure
    else:
        # load embeddings from text files
        emb1 = Embedding(path1)
        emb2 = Embedding(path2)

        if args.dist == "sem_disp":
            dist = emb2.sem_disp(other=emb1, n=args.truncate)
        elif args.dist == "eigen_overlap":
            dist = emb2.eigen_overlap(other=emb1, n=args.truncate)
        elif args.dist == 'weighted_eigen_overlap':
            dist = emb2.eigen_overlap(other=emb1, weighted=True, exp=args.exp, normalize=not args.no_norm, symmetric=args.symmetric, n=args.truncate)
        elif args.dist == 'anchor_eigen_overlap':
            assert args.truncate > 0, 'Need to use top n for anchor metric'
            print(f'Loading {anchor_path}.')
            emb1_anchor, emb2_anchor, vocab_anchor = pickle.load(open(anchor_path, 'rb'))
            dist = emb2.anchor_eigen_overlap(emb1, emb1_anchor=emb1_anchor, emb2_anchor=emb2_anchor, vocab=vocab_anchor, exp=args.exp, n=args.truncate)
        elif args.dist == 'fro_norm':
            dist = emb2.fro_norm(other=emb1, n=args.truncate)
        elif args.dist == 'pip':
            dist = emb2.pip_loss(other=emb1, n=args.truncate, random=args.random)
        elif 'knn' in args.dist:
            dist = emb2.knn(other=emb1, n=args.truncate, nquery=args.nquery, nneighbors=args.nneighbors)
        elif args.dist == 'spec_norm':
            dist = emb2.spectral_norm(other=emb1, n=args.truncate)
    return dist

def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    algos = [args.algo]

    assert args.dist != 'weighted_eigen_overlap' or args.exp is not None, "Must provide exponent for weighted eigen overlap."
    assert args.dim is not None or not args.compressed, "Must provide a dimension for compression evaluation"

    # learning rates used to train each embedding
    wiki_lrs = {'glove': 0.01, 'mc': 0.2, 'w2v_cbow': 0.05, 'pi': None, 'ft_sg': 0.05}

    seeds = [1234, 1235, 1236]

    # Set dimensions
    if args.dim:
        dims = [args.dim]
    else:
        dims = [25, 50, 100, 200, 400, 800]

    # Set precisions
    if args.bitrate:
        bitrates = [args.bitrate]
    else:
        bitrates = [1,2,4,8,16,32]

    dist_tag = get_dist_tag(args)
    results = []
    pool = mp.Pool(processes=args.nprocs)
    for algo in algos:
        # use same learning rate across dimensions
        lr = wiki_lrs[algo]
        dim = args.dim
        for bitrate in bitrates:
            seed_runs = [pool.apply_async(run_seed, args=(algo,seed,dim,lr,bitrate,args.compress_type,args)) for seed in seeds]
            seed_results = [p.get() for p in seed_runs]
            for i,seed in enumerate(seeds):
                row = {}
                row["algo"] = algo
                row["seed"] = seed
                row["dim"] = dim
                row["lr"] = lr
                row["bitrate"] = bitrate
                row[dist_tag] = seed_results[i]
                row["compress_type"] = args.compress_type
                print(row)
                results.append(row)

    # Dump results
    df_results = pd.DataFrame(results)
    if args.compressed:
        df_sum = df_results.groupby(['algo', 'bitrate']).aggregate(['mean', 'std']).reset_index()
    else:
        df_sum = df_results.groupby(['algo', 'dim']).aggregate(['mean', 'std']).reset_index()
    print(df_results)
    print(df_sum)
    dist_tag = get_final_dist_tag(dist_tag, args)
    df_results_path = f"{args.out}/{args.algo}_{dist_tag}_results.pkl"
    df_results.to_pickle(df_results_path)

if __name__ == '__main__':
    main()