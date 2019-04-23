from abc import ABC, abstractmethod

from typing import Callable, List, Tuple

import torch
from torch import Tensor

from dlkit.Loss import Loss
from dlkit.nn.modules.gan.DiscriminatorPenalty import DiscriminatorPenalty


class ConditionalDiscriminator(torch.nn.Module, ABC):

    @abstractmethod
    def forward(self, x: Tensor, condition: Tensor) -> Tensor: pass


