import numpy as np
import dlkit.data3d.datasets

import torch.utils.data
import torch.utils.data.dataloader


def default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    error_msg = "batch must contain tensors, numbers, dicts or lists or None; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        if torch.utils.data.dataloader._use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif batch[0] is None:
        assert all(item is None for item in batch)
        return None
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if torch.utils.data.dataloader.re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return torch.utils.data.dataloader.numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], torch.utils.data.dataloader.int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], torch.utils.data.dataloader.string_classes):
        return batch
    elif isinstance(batch[0], torch.utils.data.dataloader.container_abcs.Mapping):
        return {key: [d[key] for d in batch] if key.startswith('_') else default_collate([d[key] for d in batch])
                for key in batch[0]}
    elif isinstance(batch[0], torch.utils.data.dataloader.container_abcs.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


class DataLoader(torch.utils.data.dataloader.DataLoader):

    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, collate_fn=default_collate, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        super(DataLoader, self).__init__(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, batch_sampler=batch_sampler,
            num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory, drop_last=drop_last,
            timeout=timeout, worker_init_fn=worker_init_fn,
        )


def split(dataset, val=0.1, test=0., split_num=0, seed=19031):
    assert 0 <= test < 1
    assert 0 < val < 1
    rs = np.random.RandomState(seed)
    size = len(dataset)
    indices = rs.permutation(range(size))

    test_size = int(test * size)
    test_set = indices[:test_size]
    indices = indices[test_size:]

    val_size = int(val * (size - test_size))
    splits = np.array_split(indices, val_size)
    val_set = torch.utils.data.Subset(dataset, splits.pop(split_num))
    train_set = torch.utils.data.Subset(dataset, np.concatenate(splits))

    if test:
        return train_set, val_set, torch.utils.data.Subset(dataset, test_set)
    else:
        return train_set, val_set


def cycle(dataloader):
    while True:
        for sample in dataloader:
            yield sample
