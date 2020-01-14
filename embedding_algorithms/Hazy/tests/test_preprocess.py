import filecmp
import hashlib
import os
import unittest

import hazy

import hazytensor

# Test preprocess: vocab_count, and cooccur


class MainTest(unittest.TestCase):
    def test_preprocess(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        hazy.vocab_count(
            corpus_file=dir_path + "/data/sample_data.txt",
            vocab_file=dir_path + "/tmp_vocab.txt",
            min_count=5,
        )

        with open(dir_path + "/tmp_vocab.txt", "rb") as fin:
            f1_md5 = hashlib.md5(fin.read()).hexdigest()
        with open(dir_path + "/data/sample_data_vocab.txt", "rb") as fin:
            f2_md5 = hashlib.md5(fin.read()).hexdigest()

        self.assertEqual(f1_md5, f2_md5)

        print("Passed vocab count test!")
        hazy.cooccur(
            corpus_file=dir_path + "/data/sample_data.txt",
            cooccur_file=dir_path + "/tmp_cooccur.bin",
            vocab_file=dir_path + "/tmp_vocab.txt",
            memory=4.0,
            window_size=10,
        )

        result = filecmp.cmp(
            dir_path + "/tmp_cooccur.bin", dir_path + "/data/sample_data_cooccur.bin"
        )

        self.assertEqual(result, True)

        print("Passed cooccur test!")

        hazy.shuffle(
            cooccur_in=dir_path + "/tmp_cooccur.bin",
            cooccur_out=dir_path + "/tmp_cooccur_shuffle.bin",
            memory=4.0,
        )

        data = {}

        sm1 = hazy.coo_from_file(dir_path + "/tmp_cooccur.bin").scipy()
        for i, j, v in zip(sm1.row, sm1.col, sm1.data):
            data[(i, j)] = v

        sm2 = hazy.coo_from_file(dir_path + "/tmp_cooccur_shuffle.bin").scipy()

        self.assertEqual(len(sm1.data), len(sm2.data))

        for i, j, v in zip(sm2.row, sm2.col, sm2.data):
            self.assertEqual(data[(i, j)], v)

        print("Passed shuffle test!")


if __name__ == "__main__":
    unittest.main()
