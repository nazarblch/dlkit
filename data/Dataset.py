import torch
import numpy as np


class Dataset(torch.utils.data.Dataset):

    def split(dataset, num_splits=10, test=0., split_num=0, seed=19031):
        assert 0 <= test < 1
        assert num_splits
        if split_num >= num_splits:
            raise IndexError('split number must be in [0, num_splits)')
        rs = np.random.RandomState(seed)
        size = len(dataset)
        indices = rs.permutation(range(size))

        test_size = int(test * size)
        test_set = indices[:test_size]
        indices = indices[test_size:]

        splits = np.array_split(indices, num_splits)
        val_set = torch.utils.data.Subset(dataset, splits.pop(split_num))
        train_set = torch.utils.data.Subset(dataset, np.concatenate(splits))

        if test:
            return train_set, val_set, torch.utils.data.Subset(dataset, test_set)
        else:
            return train_set, val_set