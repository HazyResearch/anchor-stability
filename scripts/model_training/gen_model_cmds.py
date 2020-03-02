"""
Generate commands for model training for a list of embeddings.
"""

import argparse
import glob
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--homedir', type=str, help='Root directory of code', required=True)
    parser.add_argument('--python', type=str, help='Python version', required=True)
    parser.add_argument('--resultdir', type=str, help='Directory to save results', required=True)
    parser.add_argument('--dataset', type=str, help='Dataset for sentiment analysis')
    parser.add_argument('--model', type=str, default='la', choices=['cnn', 'lstm', 'la'])
    parser.add_argument('--task', type=str, choices=['sentiment', 'ner'], required=True)
    parser.add_argument('--gpu', type=int, help='GPU id', default=0)
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--crf', action='store_true', help='Use CRF for NER')
    parser.add_argument('--finetune', action='store_true', help='Finetune embeddings')
    parser.add_argument('--model_seed', type=int, help='Seed of the model')
    parser.add_argument('--data_seed', type=int, help='Seed of the data')
    parser.add_argument('--seed_test', action='store_true', help='Used for testing different model and data seeds')
    return parser.parse_args()

def main():
    args = parse_args()
    resultdir = args.resultdir
    homedir = args.homedir

    # read in list of embeddings from stdin
    embs = [line.strip() for line in sys.stdin.readlines()]

    os.makedirs(resultdir, exist_ok=True)
    if args.dataset == 'sst2':
        dataset = 'sst'
    else:
        dataset = args.dataset

    assert len(embs) > 0, 'Must provide embs with stdin'

    for emb in embs:
        terms = os.path.basename(emb).split('_')
        seed = terms[terms.index('seed') + 1].split('.')[0]
        dim = terms[terms.index('dim') + 1].split('.')[0]
        if args.task == 'sentiment':
            assert args.lr is not None, 'Must provide lr'
            assert args.model is not None and args.dataset is not None, 'Must provide model and dataset for sentiment task!'
            line = f'CUDA_VISIBLE_DEVICES={args.gpu} {args.python} {homedir}/scripts/model_training/train_downstream.py --dataset {dataset} --embed_path {emb} --data_path {homedir}/third_party/sentence_classification/data --seed {seed} --model {args.model} --task sentiment --lr {args.lr} --no_normalize'
            resultpath = f'{resultdir}/{os.path.basename(emb)}/{args.dataset}'
            if args.predict:
                line += ' --predict'
            if args.finetune:
                line += ' --finetune'
            if args.model_seed is not None:
                line += f' --model_seed {args.model_seed}'
            if args.data_seed is not None:
                line += f' --data_seed {args.data_seed}'
            if args.seed_test:
                line += f' --model_seed {int(seed)+1000} --data_seed {int(seed)+1000}'
            print(f'{line} --resultdir {resultpath}')

        elif args.task == 'ner':
            assert (args.lr is not None or args.predict), 'Must provide a lr for training'
            if not args.crf:
                line = f'{args.python} {homedir}/scripts/model_training/train_downstream.py --embed_path {emb} --data_path {homedir}/third_party/flair/resources/tasks --seed {seed} --resultdir {resultdir}/{os.path.basename(emb)}/ner_no_crf_lr_{args.lr} --task ner --lr {args.lr}'
            else:
                line = f'{args.python} {homedir}/scripts/model_training/train_downstream.py --embed_path {emb} --data_path {homedir}/third_party/flair/resources/tasks --seed {seed} --resultdir {resultdir}/{os.path.basename(emb)}/ner_crf_lr_{args.lr} --crf --task ner --lr {args.lr}'
            if args.predict:
                line += ' --predict'
            print(line)

if __name__ == '__main__':
    main()
