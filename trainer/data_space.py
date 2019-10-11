from typing import TypeVar, List, Generic
from torch import Tensor
import torch
import numpy as np


Shape = TypeVar("Shape", torch.Size, List[torch.Size])


class DataSpace(Generic[Shape]):
    def __init__(self, name: str, shape: Shape):
        self.name = name
        self.shape = shape


class DescartesProduct(DataSpace[List[torch.Size]]):

    def __init__(self, space1: DataSpace, space2: DataSpace):

        shape = space1.shape
        if not isinstance(shape, list):
            shape = [shape]

        if not isinstance(space2.shape, list):
            shape += [space2.shape]
        else:
            shape += space2.shape

        super(DescartesProduct, self).__init__(space1.name + " X " + space2.name, shape)


class SpaceToDataMap(Generic[Shape]):

    def __init__(self, space: DataSpace[Shape], data: List[Tensor]):

        shape = space.shape
        if not isinstance(shape, list):
            shape = [shape]

        for i, s_i in enumerate(shape):
            assert s_i == data[i]
