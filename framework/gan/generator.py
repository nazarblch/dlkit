from abc import ABC, abstractmethod
from typing import Optional, List

import torch
from torch import Tensor, nn

from framework.module import NamedModule
from .noise.Noise import Noise


class Generator(nn.Module, ABC):

    def __init__(self):
        super(Generator, self).__init__()

    @abstractmethod
    def forward(self, *noise: Tensor) -> Tensor: pass


