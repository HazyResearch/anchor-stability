import os
import sys
import argparse
import time
import random

import numpy as np
import dataloader

from utils import *
import logging
import pickle

FORMAT = '%(levelname)s|%(asctime)s|%(name)s|line_num:%(lineno)d| %(message)s'


def deep_iter(x):
    if isinstance(x, list) or isinstance(x, tuple):
        for u in x:
            for v in deep_iter(u):
                yield v
    else:
        yield x

def main(args):
    #if args.dataset == 'mr':
    #    data, label = dataloader.read_MR(args.path, seed=args.seed)
    #elif args.dataset == 'subj':
    #    data, label = dataloader.read_SUBJ(args.path, seed=args.seed)
    #elif args.dataset == 'cr':
    #    data, label = dataloader.read_CR(args.path, seed=args.seed)
    #elif args.dataset == 'mpqa':
    #    data, label = dataloader.read_MPQA(args.path, seed=args.seed)
    #elif args.dataset == 'trec':
    #    train_x, train_y, test_x, test_y = dataloader.read_TREC(args.path, seed=args.seed)
    #    data = train_x + test_x
    #    label = None
    #elif args.dataset == 'sst':
    #    train_x, train_y, valid_x, valid_y, test_x, test_y = dataloader.read_SST(args.path, seed=args.seed)
    #    data = train_x + valid_x + test_x
    #    label = None
    #else:
    #    raise Exception("unknown dataset: {}".format(args.dataset))

    train_x, _, valid_x, _, test_x, _ = dataloader.read_split_dataset(args.path, args.dataset)
    data = train_x + valid_x + test_x 
    word2id = {}
    for w in deep_iter(data):
        if w not in word2id:
            word2id[w] = len(word2id)
    logging.info("Word dict size: {}".format(len(word2id)))

    embs = dataloader.load_embedding(args.embedding, word2id)
    logging.info("Total size: {}".format(len(embs[0])))

    pickle.dump(embs, open( args.output + ".pkl" , "wb" ))

if __name__ == "__main__":

    argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    argparser.add_argument("--dataset", type=str, default="mr", help="which dataset")
    argparser.add_argument("--path", type=str, required=True, help="path to corpus directory")
    argparser.add_argument("--embedding", type=str, required=True, help="word vectors")
    argparser.add_argument("--output", type=str, required=True, help="output name")
    argparser.add_argument("--seed", type=int, default=1234)
    argparser.add_argument("--logfile", type=str, required=True, help='logfile') 
    args = argparser.parse_args()

    logging.basicConfig(format=FORMAT, filename=args.logfile, level=logging.INFO)
    # Dump command line arguments
    logging.info("Machine: " + os.uname()[1])
    logging.info("CMD: python " +  " ".join(sys.argv))
    print_key_pairs(args.__dict__.items(), title="Command Line Args", print_function=logging.info)

    main(args)
