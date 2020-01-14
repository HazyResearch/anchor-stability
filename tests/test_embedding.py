import unittest
from unittest.mock import Mock, patch

import numpy as np
import utils
from scipy.linalg import orthogonal_procrustes

from anchor.embedding import Embedding

class EmbeddingTest(unittest.TestCase):
    def test_save_load(self):
        # save embedding
        vecs = np.array([[0, 1, 2], [3, 4, 5]])
        vocab = ["cat", "dog"]
        tmpfile = "tmp.txt"
        original_emb = Embedding(vecs=vecs, vocab=vocab)
        original_emb.save(tmpfile)

        # load into a new embedding
        new_emb = Embedding(tmpfile)
        new_vecs = new_emb.m
        utils.clean_files("tmp.txt")
        np.testing.assert_array_equal(vecs, new_vecs)

    def test_align(self):
        # test basic align
        vocab = ["cat", "dog"]
        vecs1 = np.array([[0, 1, 2], [3, 4, 5]])
        vecs2 = np.array([[4, 5, 6], [7, 8, 9]])
        emb1 = Embedding(vecs=vecs1, vocab=vocab)
        emb2 = Embedding(vecs=vecs2, vocab=vocab)
        R, _ = orthogonal_procrustes(vecs2, vecs1)
        expected_vec = np.dot(vecs2, R)
        emb2.align(emb1)
        np.testing.assert_array_equal(expected_vec, emb2.m)
        np.testing.assert_array_equal(emb1, emb2.reference)

        # test align with subembeds
        vocab1 = ["cat", "dog"]
        vocab2 = ["cat", "giraffe", "dog"]
        vecs1 = np.array([[0, 1, 2], [3, 4, 5]])
        vecs2 = np.array([[4, 5, 6], [7, 8, 9], [0, 3, 4]])
        emb1 = Embedding(vecs=vecs1, vocab=vocab1)
        emb2 = Embedding(vecs=vecs2, vocab=vocab2)
        R, _ = orthogonal_procrustes(np.array([[4, 5, 6], [0, 3, 4]]), vecs1)
        expected_vec = np.dot(vecs2, R)
        emb2.align(emb1)
        np.testing.assert_array_equal(expected_vec, emb2.m)
        np.testing.assert_array_equal(emb1, emb2.reference)

    def test_get_subembed(self):
        vecs = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        vocab = ["cat", "dog", "giraffe"]
        full_emb = Embedding(vecs=vecs, vocab=vocab)
        new_vecs = full_emb.get_subembed(["giraffe", "dog"])
        expected_vecs = np.array([[6, 7, 8], [3, 4, 5]])
        np.testing.assert_array_equal(expected_vecs, new_vecs)

        new_vecs = full_emb.get_subembed(["cat"])
        expected_vecs = np.array([[0, 1, 2]])
        np.testing.assert_array_equal(expected_vecs, new_vecs)

if __name__ == "__main__":
    unittest.main()
