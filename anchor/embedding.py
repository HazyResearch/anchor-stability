import copy
import io
import logging
import numpy as np
import os
from scipy import spatial
from scipy.linalg import orthogonal_procrustes
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)

# modified from https://github.com/williamleif/histwords/blob/master/representations/embedding.py

class Embedding:
    """
    Embedding class to load embeddings and compute distances between embeddings.
    """

    def __init__(
        self, emb_path=None, vecs=None, vocab=None, header=False):
        # Use lazy loading of the embedding if not provided
        if (vecs is not None and vocab is None) or (vecs is None and vocab is not None):
            raise ValueError("vecs and vocab must be provided together")
        self.emb_path = emb_path
        self._iw = vocab
        self._wi = None
        self._dim = None
        self._m = None
        if vecs is not None:
            self._m = copy.deepcopy(vecs)
        if self._iw is not None:
            self._wi = {w: i for i, w in enumerate(self._iw)}
        if self._m is not None:
            self._dim = self._m.shape[1]
        self.header = header  # bool, whether there is a header in emb file

    @property
    def wi(self):
        if self._wi is None:
            # Load all embedding info
            self._load()
        return self._wi

    @property
    def iw(self):
        if self._iw is None:
            # Load all embedding info
            self._load()
        return self._iw

    @property
    def m(self):
        if self._m is None:
            # Load all embedding info
            self._load()
        return self._m

    @property
    def dim(self):
        if self._dim is None:
            # Load all embedding info
            self._load()
        return self._dim

    def __getitem__(self, key):
        if self.oov(key):
            raise KeyError
        else:
            return self.represent(key)

    def __iter__(self):
        return self.iw.__iter__()

    def __contains__(self, key):
        return not self.oov(key)

    def oov(self, w):
        return not (w in self.wi)

    def represent(self, w):
        if w in self.wi:
            return self.m[self.wi[w], :]
        return np.zeros(self.dim)

    def _load(self):
        """Load embeddings from file."""
        logger.info("Loading embedding...")
        f = open(self.emb_path, "r")
        dat = [_.rstrip() for _ in f]
        # Ignore the header and incomplete rows
        if len(dat[0].split()) == 2:
            dat = dat[1:]
        # ignore word
        self._dim = len(dat[0].split()) - 1
        self._m = np.zeros((len(dat), self._dim))
        vocab = []
        cnt = 0
        for i, _ in enumerate(dat):
            d = _.split(' ')
            if len(d) != self._dim + 1:
                cnt += 1
            w = ' '.join(d[:-self._dim])
            self._m[i] = d[-self._dim:]
            vocab.append(w)
        self._wi, self._iw = dict([(a, i) for i, a in enumerate(vocab)]), vocab
        if cnt > 0:
            logger.debug(f"Found {cnt} empty word(s)")
        f.close()

    def save(self, path=None):
        if path is None:
            path = self.emb_path

        assert path is not None

        # Write current vecs to file
        logger.info(f"Writing embedding to {path}")

        if os.path.exists(path):
            logger.warning(f"Overwriting existing embedding file: {path}")

        # https://github.com/facebookresearch/fastText/blob/99f23802d4572ba50417b062137fbd20aa03a794/alignment/utils.py
        n = len(self._iw)
        fout = io.open(path, "w", encoding="utf-8")
        if self.header:
            fout.write("%d %d\n" % (n, self.dim))
        for i in range(n):
            fout.write(
                self._iw[i]
                + " "
                + " ".join(map(lambda a: "%.6f" % a, self._m[i, :]))
                + "\n"
            )
        fout.close()
        self.emb_path = path

    def align(self, reference):
        if not isinstance(reference, Embedding):
            raise ValueError("Argument must be an embedding")
        shared_words = list(set(reference.iw) & set(self.iw))
        num_shared = len(shared_words)
        logger.info(f"{num_shared} words are shared with the reference matrix.")
        sub_emb1 = self.get_subembed(shared_words)
        sub_emb2 = reference.get_subembed(shared_words)
        R, _ = orthogonal_procrustes(sub_emb1, sub_emb2)
        # Rotate entire matrix, not just new words
        self._m = np.dot(self._m, R)

    def normalize(self):
        norm = np.sqrt(np.sum(self.m * self.m, axis=1))
        norm[norm == 0.0] = 1.0
        self._m = self.m / norm[:, np.newaxis]

    def get_subembed(self, word_list):
        assert isinstance(
            word_list, list
        ), "Must be list to use subembed for consistent orderings"
        keep_indices = [self.wi[word] for word in word_list]
        return self.m[keep_indices, :]

    def get_subembeds_same_vocab(self, other, n=-1, random=False, return_vocab=False):
        shared_vocab = list(set(self.iw) & set(other.iw))
        num_shared = len(shared_vocab)
        logger.info(f"{num_shared} words are shared between the embeddings.")
        if n > 0:
            if not random:
                # sort words by original order in self
                # assuming this is by frequency, then the most frequent words are first
                vocab_new = {word: self.wi[word] for word in shared_vocab}
                sorted_shared_vocab , _ = zip(*sorted(vocab_new.items(), key=lambda kv: kv[1]))
                shared_vocab = list(sorted_shared_vocab)[:n]
            else:
                print('Randomly sampling vocab')
                # choose n random words
                shared_vocab = [shared_vocab[i] for i in np.random.choice(len(shared_vocab), n, replace=False)]
        emb1 = self.get_subembed(shared_vocab)
        emb2 = other.get_subembed(shared_vocab)
        if return_vocab:
            return emb1, emb2, shared_vocab
        return emb1, emb2

    def _eigen_overlap(self, X1, X2):
        # X1 and X2 are n x d where n is the dataset size, d is the feature dimensionality
        assert X1.shape[0] == X2.shape[0]
        U1, _, _ = np.linalg.svd(X1, full_matrices=False)
        U2, _, _ = np.linalg.svd(X2, full_matrices=False)
        normalizer = max(X1.shape[1], X2.shape[1])
        return np.linalg.norm(np.matmul(U1.T, U2), ord="fro") ** 2 / normalizer

    def _weighted_eigen_overlap(self, X1, X2, exp=1, normalize=True):
        # X1 and X2 are n x d where n is the dataset size, d is the feature dimensionality
        assert X1.shape[0] == X2.shape[0]
        U1, S1, _ = np.linalg.svd(X1, full_matrices=False)
        U2, S2, _ = np.linalg.svd(X2, full_matrices=False)
        if normalize:
            normalizer = np.sum(np.diag(S2**(exp*2)))
            return np.linalg.norm(np.matmul(np.matmul(U1.T, U2), np.diag(S2**exp)), ord="fro")**2 / normalizer
        return np.linalg.norm(np.matmul(np.matmul(U1.T, U2), np.diag(S2**exp)), ord="fro")**2

    def eigen_overlap(self, other, weighted=False, exp=1, normalize=True, n=-1):
        emb1, emb2 = self.get_subembeds_same_vocab(other, n=n)
        if not weighted:
            norm = self._eigen_overlap(emb1, emb2)
        else:
            norm = self._weighted_eigen_overlap(emb1, emb2, exp=exp, normalize=normalize)
        return norm

    def eis(self, other, curr_anchor, other_anchor, vocab=None, exp=3, n=-1):
        if vocab is None:
            # get vocab from anchor embs
            curr_anchor, other_anchor, vocab = curr_anchor.get_subembeds_same_vocab(other_anchor, n=n, return_vocab=True)
        curr_emb = self.get_subembed(vocab)
        other_emb = other.get_subembed(vocab)
        V1, R1, _ = np.linalg.svd(curr_anchor, full_matrices=False)
        V2, R2, _ = np.linalg.svd(other_anchor, full_matrices=False)
        U1, _, _ = np.linalg.svd(curr_emb, full_matrices=False)
        U2, _, _ = np.linalg.svd(other_emb, full_matrices=False)
        R1_a = np.diag(R1**exp)
        R2_a = np.diag(R2**exp)

        t1 = np.linalg.norm(np.matmul(np.matmul(U1.T, V1), R1_a), ord='fro')**2
        t2 = np.linalg.norm(np.matmul(np.matmul(U2.T, V1), R1_a), ord='fro')**2
        t3_1 = np.matmul(R1_a, np.matmul(V1.T, U2))
        t3_2 = np.matmul(np.matmul(U2.T, U1), np.matmul(U1.T, V1))
        t3 = np.trace(np.matmul(np.matmul(t3_1, t3_2), R1_a))

        t4 = np.linalg.norm(np.matmul(np.matmul(U1.T, V2), R2_a), ord='fro')**2
        t5 = np.linalg.norm(np.matmul(np.matmul(U2.T, V2), R2_a), ord='fro')**2
        t6_1 = np.matmul(R2_a, np.matmul(V2.T, U2))
        t6_2 = np.matmul(np.matmul(U2.T, U1), np.matmul(U1.T, V2))
        t6 = np.trace(np.matmul(np.matmul(t6_1, t6_2), R2_a))

        normalizer = np.trace(np.diag(R1**(exp*2))) + np.trace(np.diag(R2**(exp*2)))
        return (t1 + t2 - 2*t3 + t4 + t5 - 2*t6) / normalizer


    def sem_disp(self, other, average=True, align=True, return_shared_vocab=False, n=-1):
        if not isinstance(other, Embedding):
            raise ValueError("Only Embedding objects supported.")
        emb1, emb2 = self.get_subembeds_same_vocab(other, n=n)
        if align:
            R, _ = orthogonal_procrustes(emb1, emb2)
            emb1 = np.dot(emb1, R)
        rcos_dist = np.array(
            [spatial.distance.cosine(emb1[k], emb2[k]) for k in range(emb2.shape[0])]
        )
        if average:
            rcos_dist = np.nanmean(rcos_dist, axis=0)
        if return_shared_vocab:
            return rcos_dist, shared_vocab
        return rcos_dist

    def fro_norm(self, other, n=-1):
        if not isinstance(other, Embedding):
            raise ValueError("Only Embedding objects supported.")
        emb1, emb2 = self.get_subembeds_same_vocab(other, n=n)
        return float(np.linalg.norm(emb1 - emb2, ord='fro'))

    def pip_loss(self, other, n=10000, random=False):
        assert n > 0, "Truncation required for pip loss"
        X, Xq = self.get_subembeds_same_vocab(other, n=n, random=random)
        K = X @ X.T
        Kq = Xq @ Xq.T
        return float(np.linalg.norm(K - Kq))

    def knn(self, other, n=10000, nquery=1000, nneighbors=5):
        np.random.seed(1234)
        assert n > 0 and nquery > 0 and nneighbors > 0, "N, nquery, nneighbors must be > 0"
        emb1, emb2 = self.get_subembeds_same_vocab(other, n=n)

        # randomly sample queries from n
        rand_indices = np.random.choice(n, nquery, replace=False)
        query1 = emb1[rand_indices]
        query2 = emb2[rand_indices]

        neigh1 = NearestNeighbors(nneighbors+1, metric='cosine')
        neigh1.fit(emb1)
        _, neighbors1 = neigh1.kneighbors(X=query1)

        neigh2 = NearestNeighbors(nneighbors+1, metric='cosine')
        neigh2.fit(emb2)
        _, neighbors2 = neigh2.kneighbors(X=query2)

        def _filter_nn(neighbors):
            actual_neighbors = np.zeros((len(neighbors), nneighbors))
            for i, nn in enumerate(neighbors):
                nn = np.array(nn)
                # Delete query itself from neighbors
                try:
                    actual_neighbors[i] = np.delete(nn, np.where(nn == rand_indices[i]))
                # Cut last neighbor if query not in list
                except:
                    actual_neighbors[i] = nn[:-1]
            return actual_neighbors

        neighbors1 = _filter_nn(neighbors1)
        neighbors2 = _filter_nn(neighbors2)

        assert neighbors1.shape[1] == nneighbors and neighbors2.shape[1] == nneighbors, 'Dimensions not correct for nearest neighbors'
        count = 0.
        for n1, n2 in zip(neighbors1, neighbors2):
            count += len(set(n1).intersection(n2)) / len(n1)
        return count / len(neighbors1)