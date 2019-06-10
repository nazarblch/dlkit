from abc import ABC, abstractmethod
from typing import Callable, List, TypeVar, Generic

import torch
from torch import Tensor


class Discriminator(torch.nn.Module, ABC):

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor: pass










