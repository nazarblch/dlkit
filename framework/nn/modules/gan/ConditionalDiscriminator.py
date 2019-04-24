from abc import ABC, abstractmethod
from typing import Callable

import torch
from torch import Tensor

from framework.nn.modules.gan.Discriminator import Discriminator


class DiscriminatorObject(Discriminator):

    def __init__(self, condition: Tensor, apply: Callable[[Tensor, Tensor], Tensor]):
        super().__init__()
        self.condition = condition
        self.apply = apply

    def forward(self, x: Tensor) -> Tensor:
        return self.apply(x, self.condition)


class ConditionalDiscriminator(torch.nn.Module, ABC):

    @abstractmethod
    def forward(self, x: Tensor, condition: Tensor) -> Tensor: pass

    def discriminator(self, condition: Tensor) -> Discriminator:
        return DiscriminatorObject(condition, self.forward)





