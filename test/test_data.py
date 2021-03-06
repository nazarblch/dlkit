import unittest
import numpy as np
from dlkit.data_loader.DataLoader import DataLoader


from dlkit.data_loader.Dataset import Dataset


class Dummy(Dataset):

    def __init__(self, seq):
        self.seq = seq

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, i):
        return self.seq[i]


def to_set(dataset):
    return {dataset[i] for i in range(len(dataset))}


class TestSplit(unittest.TestCase):

    def assertEmpty(self, s, msg=None):
        self.assertEqual(s, set(), msg)

    def test_reproducibility(self):
        dataset = Dummy(np.arange(10))
        train, val, test = dataset.split(num_splits=9, test=0.1, split_num=0, seed=94)
        self.assertEqual(to_set(train), {7, 8, 1, 9, 3, 5, 0, 2})
        self.assertEqual(to_set(val), {6})
        self.assertEqual(to_set(test), {4})

    def test_splits(self):
        size = 123
        num_splits = 10
        test_frac = 0.1
        seed = 19

        test_size = int(size * test_frac)
        dataset = Dummy(np.arange(size))
        all_samples = to_set(dataset)
        results = []
        for i in range(num_splits):
            train, valid, test = dataset.split(num_splits=num_splits, test=test_frac, split_num=i, seed=seed)
            train, valid, test = to_set(train), to_set(valid), to_set(test)
            # check that train, valid and test splits do not intersect
            self.assertEmpty(train & valid & test)
            # check that train, valid and test splits together contain all samples
            self.assertEqual(train | valid | test, all_samples)
            results.append((train, valid, test))
        train, valid, test = zip(*results)
        # check that all valid splits do not intersect
        self.assertEmpty(set.intersection(*valid))
        # check test split size
        self.assertEqual(len(test[0]), test_size)
        # check that all test splits are the same
        self.assertTrue(all(s == test[0] for s in test))

    def test_max_split_num(self):
        dataset = Dummy(100)
        with self.assertRaises(IndexError):
            dataset.split(num_splits=10, split_num=10)


class TestLoader(unittest.TestCase):

    def test_none_all(self):
        dataset = Dummy([{'x': None}, {'x': None}])
        loader = iter(DataLoader(dataset, 2))
        self.assertDictEqual(next(loader), {'x': None})

    def test_none_some(self):
        dataset = Dummy([{'x': 1}, {'x': None}])
        loader = iter(DataLoader(dataset, 2))
        with self.assertRaises(TypeError):
            next(loader)

    def test_meta(self):
        dataset = Dummy([{'_x': 1}, {'_x': None}])
        loader = iter(DataLoader(dataset, 2))
        self.assertDictEqual(next(loader), {'_x': [1, None]})


if __name__ == '__main__':
    unittest.main()
