import os
import unittest

import hazy

import hazytensor

# Test coo_from_file function


class MainTest(unittest.TestCase):
    def test_coo_from_file(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        n_words = 10
        # Loads the output cooccur from GloVe into a scipy COO.
        sm1 = hazy.coo_from_file(dir_path + "/data/sample_data_1.bin").scipy()
        for i, j, v in zip(sm1.row, sm1.col, sm1.data):
            self.assertEqual(1. * i * n_words + j, v)

        sm2 = hazy.coo_from_file(dir_path + "/data/sample_data_2.bin").scipy()
        for i, j, v in zip(sm2.row, sm2.col, sm2.data):
            self.assertEqual(1. * ((i * n_words + j) % 13), v)


if __name__ == "__main__":
    unittest.main()
