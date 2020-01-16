import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import logging

def deep_iter(x):
    if isinstance(x, list) or isinstance(x, tuple):
        for u in x:
            for v in deep_iter(u):
                yield v
    else:
        yield x

class CNN_Text(nn.Module):

    def __init__(self, n_in, widths=[3,4,5], filters=100):
        super(CNN_Text,self).__init__()
        Ci = 1
        Co = filters
        h = n_in
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (w, h)) for w in widths])

    def forward(self, x):
        # x is (batch, len, d)
        x = x.unsqueeze(1) # (batch, Ci, len, d)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(batch, Co, len), ...]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]
        x = torch.cat(x, 1)
        return x


class EmbeddingLayer(nn.Module):
    def __init__(self, n_d, words, embs=None, fix_emb=True, oov='<oov>', pad='<pad>', normalize=True, num_pad=0, project_dim=0, normalize_list=None, median=1, num_normalize=0):
        super(EmbeddingLayer, self).__init__()
        word2id = {}
        if embs is not None:
            embwords, embvecs = embs
            for word in embwords:
                if word in word2id:
                    raise ValueError(f"Duplicate words in pre-trained embeddings, {word}")
                word2id[word] = len(word2id)

            logging.info("{} pre-trained word embeddings loaded.".format(len(word2id)))
            if n_d != len(embvecs[0]):
                logging.warn("n_d ({}) != word vector size ({}). Use {} for embeddings.".format(
                    n_d, len(embvecs[0]), len(embvecs[0])
                ))
                n_d = len(embvecs[0])

        for w in deep_iter(words):
            if w not in word2id:
                word2id[w] = len(word2id)

        if oov not in word2id:
            word2id[oov] = len(word2id)

        if pad not in word2id:
            word2id[pad] = len(word2id)

        self.word2id = word2id
        self.n_V, self.n_d = len(word2id), n_d
        if embs is not None:
            logging.info("Number of vectors: {}, Number of loaded vectors: {}, Number of oov {}".format(
                self.n_V, len(embwords), self.n_V - len(embwords)))
            print(f"Number of vectors: {self.n_V}, Number of loaded vectors: {len(embwords)}, Number of oov {self.n_V - len(embwords)}")
        self.oovid = word2id[oov]
        self.padid = word2id[pad]
        self.embedding = nn.Embedding(self.n_V, n_d)
        self.embedding.weight.data.uniform_(-0.25, 0.25)
        self.project_dim = project_dim

        if embs is not None:
            weight  = self.embedding.weight
            weight.data[:len(embwords)].copy_(torch.from_numpy(embvecs))
            logging.info("embedding shape: {}".format(weight.size()))

        if project_dim > 0:
            # random projection matrix
            projection = torch.from_numpy(np.random.uniform(-1./np.sqrt(n_d), 1./np.sqrt(n_d), size=(project_dim,n_d))).float()
            self.embedding.weight.data = self.embedding.weight.data.matmul(projection.t())
            logging.info("Generated projection matrix.")
            # update dimensions of embedding layer
            self.n_d = project_dim

        if num_pad > 0:
            logging.info("Padding embedding to {}".format(num_pad))
            old_weight = self.embedding.weight.data
            # copy weight data into new self.embedding
            self.embedding = nn.Embedding(self.n_V, num_pad)

            # set values to zero
            weight = self.embedding.weight
            weight.data.fill_(0)
            weight.data[:,:n_d].copy_(old_weight)
            # update dimensions of embedding layer
            self.n_d = num_pad

        if normalize_list is not None:
            normalize_list = open(normalize_list, 'r')
            normalize_list = [line.strip() for line in normalize_list.readlines()]
            assert len(set(normalize_list)) == num_normalize, "Number to normalize and normalize list don't match. Make sure you have specified num_normalize"
            norms = weight.data.norm(2,1)
            # normalize selected words to the median value
            # TODO (mleszczy): make vectorized
            normalize_list_set = set(normalize_list)
            # print(len(normalize_list_set))
            for word in word2id:
                if word in normalize_list_set:
                    idx = word2id[word]
                    weight.data[idx] = weight.data[idx].div_(norms[idx]/float(median))
                    # print("norm: ", np.linalg.norm(weight.data[idx].numpy()))

        if normalize:
            weight = self.embedding.weight
            norms = weight.data.norm(2,1)
            if norms.dim() == 1:
                norms = norms.unsqueeze(1)
            weight.data.div_(norms.expand_as(weight.data))

        if fix_emb:
            logging.info("Fixing embedding!")
            self.embedding.weight.requires_grad = False

    def forward(self, input):
        # if self.project_dim > 0:
        #     emb = self.embedding(input)
        #     # random projection
        #     emb = emb.matmul(self.projection.t())
        #     return emb
        return self.embedding(input)
