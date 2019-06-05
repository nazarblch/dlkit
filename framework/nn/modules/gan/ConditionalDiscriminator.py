from abc import ABC, abstractmethod
from typing import Callable

import torch
from torch import Tensor

from framework.nn.modules.gan.Discriminator import Discriminator


class ConditionalDiscriminator(torch.nn.Module, ABC):

    @abstractmethod
    def forward(self, x: Tensor, condition: Tensor) -> Tensor: pass




