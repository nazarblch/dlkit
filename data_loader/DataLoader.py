from typing import Iterator

import torch
from torch import Tensor
from torch._six import int_classes, string_classes, container_abcs
from torch.utils.data import DataLoader as DL, dataloader


class DataLoader(DL):

    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, collate_fn=None, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        if collate_fn is None:
            collate_fn = DataLoader.default_collate
        super(DataLoader, self).__init__(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, batch_sampler=batch_sampler,
            num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory, drop_last=drop_last,
            timeout=timeout, worker_init_fn=worker_init_fn,
        )

    @staticmethod
    def default_collate(batch):
        r"""Puts each data_loader field into a tensor with outer dimension batch size"""

        error_msg = "batch must contain tensors, numbers, dicts or lists or None; found {}"
        elem_type = type(batch[0])
        if isinstance(batch[0], torch.Tensor):
            out = None
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
        elif isinstance(batch[0], int_classes):
            return torch.LongTensor(batch)
        elif isinstance(batch[0], float):
            return torch.DoubleTensor(batch)
        elif isinstance(batch[0], string_classes):
            return batch
        elif isinstance(batch[0], container_abcs.Mapping):
            return {key: [d[key] for d in batch] if key.startswith('_') else DataLoader.default_collate([d[key] for d in batch])
                    for key in batch[0]}
        elif isinstance(batch[0], container_abcs.Sequence):
            transposed = zip(*batch)
            return [DataLoader.default_collate(samples) for samples in transposed]

        raise TypeError((error_msg.format(type(batch[0]))))

