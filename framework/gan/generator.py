from abc import ABC, abstractmethod
from typing import Optional, List

import torch
from torch import Tensor

from .noise.Noise import Noise


class Generator(torch.nn.Module, ABC):

    def __init__(self, noise: Noise):
        super(Generator, self).__init__()
        self.noise_gen = noise

    @abstractmethod
    def _forward_impl(self, noise: Tensor) -> Tensor: pass

    def forward(self, size: int, noise: Optional[Tensor] = None) -> Tensor:
        z: Tensor = self.noise_gen.sample(size) if noise is None else noise
        return self._forward_impl(z)
