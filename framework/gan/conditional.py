from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor

from framework.gan.discriminator import Discriminator
from framework.gan.generator import Generator
from framework.gan.noise.Noise import Noise


class ConditionalGenerator(Generator):

    def __init__(self, noise: Noise):
        super(ConditionalGenerator, self).__init__()
        self.noise_gen = noise

    @abstractmethod
    def _forward_impl(self, condition: Tensor, *noize: Tensor) -> Tensor: pass

    def forward(self,  condition: Tensor, *noize: Tensor) -> Tensor:
        return self._forward_impl(condition, *noize)


class ConditionalDiscriminator(Discriminator):

    @abstractmethod
    def forward(self, x: Tensor, *condition: Tensor) -> Tensor: pass




