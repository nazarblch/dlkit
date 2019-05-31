from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor

from framework.nn.modules.gan.noise.Noise import Noise


class ConditionalGenerator(torch.nn.Module, ABC):

    def __init__(self, noise: Noise):
        super(ConditionalGenerator, self).__init__()
        self.noise_gen = noise

    @abstractmethod
    def _forward_impl(self, noise: Tensor, condition: Tensor) -> Tensor: pass

    def forward(self, condition: Tensor, noise: Optional[Tensor] = None) -> Tensor:
        z: Tensor = self.noise_gen.sample(condition.size(0)) if noise is None else noise
        return self._forward_impl(z.to(condition.device), condition)


