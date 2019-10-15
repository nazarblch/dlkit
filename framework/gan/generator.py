from abc import ABC, abstractmethod
from typing import Optional, List

import torch
from torch import Tensor

from .noise.Noise import Noise


class Generator(torch.nn.Module, ABC):

    def __init__(self):
        super(Generator, self).__init__()

    @abstractmethod
    def _forward_impl(self, *noise: Tensor) -> Tensor: pass

    def forward(self, *noise: Tensor) -> Tensor:
        return self._forward_impl(*noise)

