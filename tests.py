import unittest
from utils import load_dataset


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.n_train = 40025
        self.batch_size = 64

    def test_load_dataset(self):
        train, val = load_dataset(self.n_train, self.batch_size)
        q, r = self.n_train // self.batch_size, self.n_train % self.batch_size
        assert len(train) == q + (r > 0)
        assert len(val) == 1
