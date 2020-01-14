import os
import sys
import argparse
import time
import pickle
import random
import math
from subprocess import check_output
import hashlib
import shutil

import numpy as np
import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# import cuda_functional as MF
sys.path.append(os.path.dirname(__file__))
import dataloader
import modules
from sentutils import *
sys.path.remove(os.path.dirname(__file__))
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from modules import deep_iter

FORMAT = '%(levelname)s|%(asctime)s|%(name)s|line_num:%(lineno)d| %(message)s'

class Model(nn.Module):
    def __init__(self, args, emb_layer, nclasses=2):
        super(Model, self).__init__()
        self.args = args
        self.drop = nn.Dropout(args.dropout)
        self.emb_layer = emb_layer
        print(emb_layer.n_d)
        if args.cnn:
            self.encoder = modules.CNN_Text(
                emb_layer.n_d,
                widths = [3,4,5]
            )
            d_out = 300
        elif args.lstm:
            self.encoder = nn.LSTM(
                emb_layer.n_d,
                args.d,
                args.depth,
                dropout = args.dropout,
            )
            d_out = args.d
        elif args.la:
            d_out = emb_layer.n_d
        self.out = nn.Linear(d_out, nclasses)

    def forward(self, input):
        if self.args.cnn:
            input = input.t()
        emb = self.emb_layer(input)
        if not self.args.la:
            emb = self.drop(emb)
        if self.args.cnn:
            output = self.encoder(emb)
        elif self.args.lstm:
            output, hidden = self.encoder(emb)
            output = output[-1]
        else:
            output = emb.sum(dim=0) / emb.size()[0]
        if not self.args.la:
            output = self.drop(output)
        return self.out(output)

def eval_model(model, valid_x, valid_y, pred_file=None, prob_file=None, sentence_emb_file=None, gold_file=None):
    model.eval()
    N = len(valid_x)
    criterion = nn.CrossEntropyLoss()
    correct = 0.0
    cnt = 0.0
    total_loss = 0.0
    preds = []
    probs = []

    if sentence_emb_file is not None:
        sentence_embs = []
        def hook(module, input, output):
            sentence_emb = output.sum(dim=0) / output.size()[0]
            sentence_embs.append(sentence_emb)

        for name, module in model.named_modules():
            if 'emb_layer.embedding' in name:
                handle = module.register_forward_hook(hook)

    gold = []
    for x, y in zip(valid_x, valid_y):
        x, y = Variable(x, volatile=True), Variable(y)
        output = model(x)
        loss = criterion(output, y)
        if torch.__version__ >= '0.4':
            total_loss += loss.data*x.size(1)
        else:
            total_loss += loss.data[0]*x.size(1)
        pred = output.data.max(1)[1]
        correct += pred.eq(y.data).cpu().sum()
        gold.append(y.data.cpu())
        cnt += y.numel()

        preds += pred.cpu().numpy().tolist()
        probs += output.data.cpu().numpy().tolist()

    if pred_file is not None:
        with open(pred_file, 'wb') as outfile:
            pickle.dump(preds, outfile)

    if prob_file is not None:
        with open(prob_file, 'wb') as outfile:
            pickle.dump(probs, outfile)

    if sentence_emb_file is not None:
        with open(sentence_emb_file, 'wb') as outfile:
            pickle.dump(sentence_embs, outfile)
        with open(gold_file, 'wb') as outfile:
            pickle.dump(gold, outfile)

    if sentence_emb_file is not None:
        handle.remove()
    model.train()
    if torch.__version__ >= '0.4':
        return 1.0-float(correct)/float(cnt)
    else:
        return 1.0-correct/cnt


def train_model(epoch, model, optimizer,
        train_x, train_y, valid_x, valid_y,
        test_x, test_y,
        best_valid, test_err,
        save_mdl=None,
        pred_file=None,
        prob_file=None):
    model.train()
    args = model.args
    N = len(train_x)
    niter = epoch*len(train_x)
    criterion = nn.CrossEntropyLoss()

    cnt = 0
    for x, y in zip(train_x, train_y):
        niter += 1
        cnt += 1
        model.zero_grad()
        x, y = Variable(x), Variable(y)
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        # logger.info("Epoch={} iter={} lr={:.6f} train_loss={:.6f}".format(
        #     epoch, niter,
        #     optimizer.param_groups[0]['lr'],
        #     loss.data
        # ))

    valid_err = eval_model(model, valid_x, valid_y)

    if torch.__version__ >= "0.4":
        logger.info("Epoch={} iter={} lr={:.6f} train_loss={:.6f} valid_err={:.6f}".format(
            epoch, niter,
            optimizer.param_groups[0]['lr'],
            loss.data,
            valid_err
        ))
    else:
        logger.info("Epoch={} iter={} lr={:.6f} train_loss={:.6f} valid_err={:.6f}".format(
            epoch, niter,
            optimizer.param_groups[0]['lr'],
            loss.data[0],
            valid_err
        ))

    if valid_err < best_valid:
        best_valid = valid_err
        test_err = eval_model(model, test_x, test_y, pred_file=pred_file, prob_file=prob_file)
        # Save model
        if save_mdl is not None:
            torch.save(model, save_mdl)

    return best_valid, test_err

def cyclic_lr(initial_lr, iteration, epoch_per_cycle):
    return initial_lr * (math.cos(math.pi * iteration / epoch_per_cycle) + 1) / 2

def main(args):
    train_x, train_y, valid_x, valid_y, test_x, test_y = dataloader.read_split_dataset(args.path, args.dataset)
    data = train_x + valid_x + test_x

    word2id = None
    # optimization for fine-tuning -- don't use full embedding
    if args.finetune:
        word2id = {}
        for w in deep_iter(data):
            if w not in word2id:
                word2id[w] = len(word2id)

    if args.embedding:
        logger.info("Using single embedding file.")
        emb_layer = modules.EmbeddingLayer(
            args.d, data,
            embs = dataloader.load_embedding(args.embedding, word2id),
            normalize=not args.no_normalize,
            num_pad=args.num_pad,
            project_dim=args.project_dim,
            normalize_list=args.normalize_list,
            median=args.median,
            num_normalize=args.num_normalize,
            fix_emb=not args.finetune
        )
    elif args.embedding_list:
        logger.info("Using embedding list.")
        embedding_list = []
        with open(args.embedding_list, 'r') as f:
            lines = f.readlines()
            assert len(lines) >= args.cycles
            for i, emb in enumerate(lines):
                logger.info("Embedding file: {embfile}".format(embfile=emb.strip()))
                embedding_list.append(modules.EmbeddingLayer(
                                    args.d, data,
                                    embs = dataloader.load_embedding(emb.strip()),
                                    normalize=not args.no_normalize).cuda())
        emb_layer = embedding_list[0]
        print("Embedding list length", len(embedding_list))
    else:
        emb_layer = modules.EmbeddingLayer(
            args.d, data,
            normalize=not args.no_normalize,
            num_pad=args.num_pad,
            project_dim=args.project_dim,
        )

    orig_emb_layer = emb_layer

    nclasses = max(train_y)+1
    logger.info(str(nclasses) + " classes in total")

    train_x, train_y = dataloader.create_batches(
        train_x, train_y,
        args.batch_size,
        emb_layer.word2id,
        sort = 'sst' in args.dataset,
        shuffle = args.data_seed != args.seed,
        # sort = args.dataset == 'sst'
		seed = args.data_seed
    )
    valid_x, valid_y = dataloader.create_batches(
        valid_x, valid_y,
        args.batch_size,
        emb_layer.word2id,
        sort = 'sst' in args.dataset,
        # use the same seed as the model for test and validation

        # change to model_seed for data
		seed = args.seed
        # sort = args.dataset == 'sst'
    )
    test_x, test_y = dataloader.create_batches(
        test_x, test_y,
        args.batch_size,
        emb_layer.word2id,
        sort = 'sst' in args.dataset,
        # use the same seed as the model for test and validation
		seed = args.seed
        # sort = args.dataset == 'sst'
    )

    if args.load_mdl is None:
        model = Model(args, emb_layer, nclasses).cuda()
    else:
        # Note: this will overwrite all parameters
        model = torch.load(args.load_mdl).cuda()

    need_grad = lambda x: x.requires_grad
    optimizer = optim.Adam(
        filter(need_grad, model.parameters()),
        lr = args.lr
    )

    best_valid = 1e+8
    test_err = 1e+8

    pred_file = os.path.join(args.out, "{tag}.pred".format(tag=args.tag))
    prob_file = os.path.join(args.out, "{tag}.prob".format(tag=args.tag))

    save_mdl = os.path.join("/tmp", os.path.basename(args.embedding), os.path.basename(args.out), "{tag}.ckpt".format(tag=args.tag)) if not args.save_mdl else args.save_mdl
    os.makedirs(os.path.dirname(save_mdl), exist_ok=True)
    # save_mdl = os.path.join(args.out, "{tag}.ckpt".format(tag=args.tag)) if not args.save_mdl else args.save_mdl

    if args.eval:
        # sentence_emb_file = os.path.join(args.out, "{tag}.sentence_emb".format(tag=args.tag))
        # gold_file = os.path.join(args.out, "{tag}.gold".format(tag=args.tag))
        # test_err = eval_model(model, test_x, test_y, pred_file=pred_file, prob_file=prob_file, sentence_emb_file=sentence_emb_file, gold_file=gold_file)

        # dump val predictions
        val_pred_file = os.path.join(args.out, "{tag}.val.pred".format(tag=args.tag))
        val_prob_file = os.path.join(args.out, "{tag}.val.prob".format(tag=args.tag))
        best_valid = eval_model(model, valid_x, valid_y, pred_file=val_pred_file, prob_file=val_prob_file)
        # don't redump test predictions
        test_err = eval_model(model, test_x, test_y)
        logger.info("=" * 40)
        logger.info("best_valid: {:.6f}".format(
            best_valid
        ))
        logger.info("test_err: {:.6f}".format(
            test_err
        ))
        logger.info("=" * 40)
        return best_valid, test_err

    vocab_file = os.path.join(args.out, "{tag}.vocab.pkl".format(tag=args.tag))
    # with open(vocab_file, 'wb') as f:
    #     pickle.dump((model.emb_layer.word2id, model.emb_layer.embedding.weight.data), f)

    # Normal training
    if not args.snapshot:
        for epoch in range(args.max_epoch):
            best_valid, test_err = train_model(epoch, model, optimizer,
                train_x, train_y,
                valid_x, valid_y,
                test_x, test_y,
                best_valid, test_err,
                save_mdl=save_mdl,
                pred_file=pred_file,
                prob_file=prob_file
            )
            if args.lr_decay>0:
                optimizer.param_groups[0]['lr'] *= args.lr_decay

    logger.info("=" * 40)
    logger.info("best_valid: {:.6f}".format(
        best_valid
    ))
    logger.info("test_err: {:.6f}".format(
        test_err
    ))
    logger.info("=" * 40)

    shutil.move(save_mdl, "{resultdir}/{tag}.ckpt".format(resultdir=args.out,tag=args.tag))

    return best_valid, test_err

def train_sentiment(cmdline_args):
    argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    argparser.add_argument("--cnn", action='store_true', help="whether to use cnn")
    argparser.add_argument("--lstm", action='store_true', help="whether to use lstm")
    argparser.add_argument("--la", action='store_true', help="whether to use la")
    argparser.add_argument("--no_normalize", action='store_true', help="Do not normalize embeddings")
    argparser.add_argument("--dataset", type=str, default="mr", help="which dataset")
    argparser.add_argument("--path", type=str, required=True, help="path to corpus directory")
    argparser.add_argument("--embedding", type=str, help="word vectors")
    argparser.add_argument("--batch_size", "--batch", type=int, default=32)
    argparser.add_argument("--max_epoch", type=int, default=100)
    argparser.add_argument("--d", type=int, default=128)
    argparser.add_argument("--dropout", type=float, default=0.5)
    argparser.add_argument("--depth", type=int, default=2)
    argparser.add_argument("--lr", type=float, default=0.001)
    argparser.add_argument("--lr_decay", type=float, default=0.0)
    # argparser.add_argument("--cv", type=int, default=0)
    argparser.add_argument("--model_seed", type=int, default=1234)
    argparser.add_argument("--data_seed", type=int, default=1234)
    argparser.add_argument("--seed", type=int, default=1234, help='General seed to support testing model and data seeds separately')
    argparser.add_argument("--save_mdl", type=str, default=None, help="Save model to this file.")
    argparser.add_argument("--load_mdl", type=str, default=None, help="Load model from this file.")
    argparser.add_argument("--out", type=str, help="Path to output directory", required=True)
    argparser.add_argument("--snapshot", action='store_true', help="Use snapshot ensembling")
    argparser.add_argument("--cycles", type=int, help="Number of cycles/snapshots to take")
    argparser.add_argument("--embedding_list", type=str, help="List of word vector files")
    argparser.add_argument("--tag", type=str, help="Tag for naming files")
    argparser.add_argument("--no_cudnn", action="store_true", help="Turn off cuDNN for deterministic CNN")
    argparser.add_argument("--num_pad", type=int, default=0, help='Number of dimensions to pad with zero')
    argparser.add_argument("--project_dim", type=int, default=0, help='Number of dimensions to randomly project embedding to.')
    argparser.add_argument("--eval", action='store_true', help="Just evaluate trained model")
    argparser.add_argument("--normalize_list", type=str, help='List of vocab words to normalize to median value.')
    argparser.add_argument("--median", type=float, help='Median value to normalize embedding vectors to.')
    argparser.add_argument("--num_normalize", type=int, help='Number to normalize. Should match normalize list contents.')
    argparser.add_argument("--finetune", action='store_true', help='Finetune embeddings')

    # argparser.add_argument("--no_cv", action="store_true", help="Merge train and validation dataset.")
    print(cmdline_args)
    if cmdline_args != '':
        args = argparser.parse_args(cmdline_args)
    else:
        args = argparser.parse_args()

    if not args.tag:
        if args.cnn:
            args.tag = "model_cnn_dropout_{dropout}_seed_{seed}_data_{seed2}_lr_{lr}".format(
                dropout=args.dropout, seed=args.model_seed, lr=args.lr, seed2=args.data_seed)
        elif args.la:
            args.tag = "model_la_dropout_{dropout}_seed_{seed}_data_{seed2}_lr_{lr}".format(
                dropout=args.dropout, seed=args.model_seed, lr=args.lr, seed2=args.data_seed)
        elif args.lstm:
            args.tag = "model_lstm_dropout_{dropout}_seed_{seed}_data_{seed2}_lr_{lr}".format(
                dropout=args.dropout, seed=args.model_seed, lr=args.lr, seed2=args.data_seed)

        if args.eval:
            args.tag += "_eval"
    else: # enables deprecated use of naming
        args.tag = args.tag

    fh = logging.FileHandler('{out}/{tag}.log'.format(out=args.out, tag=args.tag))
    formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Dump git hash
    # h = check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()
    # logger.info("Git hash: " + h)

    # Dump embedding hash
    if args.embedding:
        embedding_hash = hashlib.md5(open(args.embedding, 'rb').read()).hexdigest()
        logger.info("Embedding hash: " + embedding_hash)

    if args.no_cudnn:
        torch.backends.cudnn.enabled = False
        logger.info("CuDNN Disabled!!!")

    # Set random seed for torch
    torch.manual_seed(args.model_seed)
    torch.cuda.manual_seed(args.model_seed)

    # Set random seed for numpy
    np.random.seed(seed=args.model_seed)

    # Dump command line arguments
    logger.info("Machine: " + os.uname()[1])
    logger.info("CMD: python " +  " ".join(sys.argv))
    print_key_pairs(args.__dict__.items(), title="Command Line Args", print_function=logger.info)
    # print (args)
    return main(args)

if __name__ == "__main__":
    train_sentiment('')
