import argparse
import logging
import glob
import numpy as np
import os
import random
import sys
import torch

from third_party.sentence_classification.train_classifier import train_sentiment
from third_party.flair.ner import train_ner, eval_ner

# Downstream model training/prediction on top of an embedding
# Called from gen_model_cmds

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--embed_path", type=str, required=True)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model", type=str)
    parser.add_argument("--resultdir", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--no_normalize", action='store_true', help="Do not normalize embeddings")
    parser.add_argument("--dim", type=int)
    parser.add_argument("--predict", action='store_true')
    parser.add_argument("--lr", type=float)
    parser.add_argument("--crf", action="store_true", help="Use CRF for NER")
    parser.add_argument("--finetune", action='store_true', help='Finetune embeddings')
    parser.add_argument("--model_seed", type=int, help="Seed of the model")
    parser.add_argument("--data_seed", type=int, help="Seed of the data")
    return parser.parse_args()

def evaluate_ner(embed_path, resultdir, datadir, use_crf, lr):
    train_ner(embed_path, resultdir, datadir, use_crf, lr)

def predict_ner(embed_path, resultdir, datadir, use_crf):
    eval_ner(embed_path, resultdir, datadir, use_crf)

def evaluate_sentiment(
    embed_path, data_path, result_dir, seed, model, dataset="sst", epochs=100, lr=0.001,
    no_normalize=False, load_mdl=None, finetune=False, data_seed=None, model_seed=None
):
    # use same seed if not provided
    if data_seed is None:
        data_seed = seed
    if model_seed is None:
        model_seed = seed
    cmdlines = [
        "--dataset",
        dataset,
        "--path",
        data_path + "/",
        "--embedding",
        embed_path,
        "--max_epoch",
        str(epochs),
        "--model_seed",
        str(model_seed),
        "--data_seed",
        str(data_seed),
        "--seed",
        str(seed),
        "--lr",
        str(lr),
        "--out",
        str(result_dir)
    ]
    if model == "la":
        cmdlines += ["--la"]
    elif model == "cnn":
        cmdlines += ["--cnn"]
    elif model == "lstm":
        cmdlines += ["--lstm"]
    if no_normalize:
        cmdlines += ["--no_normalize"]
    if load_mdl is not None:
        cmdlines += ['--load_mdl', load_mdl, '--eval']
    if finetune:
        cmdlines += ['--finetune']
    err_valid, err_test = train_sentiment(cmdlines)

def main():
    args = parse_args()
    # set seeds -- need to set again per app if otherwise changed to defaults in apps
    seed = args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic=True
    np.random.seed(seed)
    random.seed(seed)

    if args.resultdir is not None:
        os.makedirs(args.resultdir, exist_ok=True)
    if args.task == 'sentiment':
        if not args.predict:
            evaluate_sentiment(
                embed_path=args.embed_path,
                data_path=args.data_path,
                seed=args.seed,
                model=args.model,
                result_dir=args.resultdir,
                dataset=args.dataset,
                no_normalize=args.no_normalize,
                lr=args.lr,
                finetune=args.finetune,
                model_seed=args.model_seed,
                data_seed=args.data_seed
            )
        else:
            pretrained = glob.glob(f'{args.resultdir}/*ckpt')
            assert len(pretrained) == 1, "More than one model available"
            pretrained = pretrained[0]
            print(pretrained)
            evaluate_sentiment(
                embed_path=args.embed_path,
                data_path=args.data_path,
                seed=args.seed,
                model=args.model,
                result_dir=args.resultdir,
                dataset=args.dataset,
                no_normalize=args.no_normalize,
                lr=args.lr,
                load_mdl=pretrained,
                model_seed=args.model_seed,
                data_seed=args.data_seed
            )
    elif args.task == 'ner':
        if args.predict:
            predict_ner(embed_path=args.embed_path,
                resultdir=args.resultdir,
                datadir=args.data_path,
                use_crf=args.crf)
        else:
            evaluate_ner(embed_path=args.embed_path,
                resultdir=args.resultdir,
                datadir=args.data_path,
                use_crf=args.crf,
                lr=args.lr)

if __name__ == "__main__":
    main()
