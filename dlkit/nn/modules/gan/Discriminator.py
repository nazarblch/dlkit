from abc import ABC, abstractmethod
from typing import Callable, List, TypeVar, Generic

import torch
from torch import Tensor

from dlkit.Loss import Loss
from dlkit.nn.modules.gan.DiscriminatorPenalty import DiscriminatorPenalty


class Discriminator(torch.nn.Module, ABC):

    __penalties: List[DiscriminatorPenalty]
    __losses: List[Callable[[Tensor, Tensor], Loss]]

    def add_penalty(self, pen: DiscriminatorPenalty) -> None:
        self.__penalties.append(pen)

    def add_pair_loss(self, loss: Callable[[Tensor, Tensor], Loss]) -> None:
        self.__losses.append(loss)

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor: pass

    def loss(self, x: Tensor, y: Tensor) -> Loss:
        Dx = self.forward(x.detach())
        Dy = self.forward(y.detach())

        alpha = torch.rand(x.size(0), *((1,) * (x.ndimension() - 1)), device=x.device)
        hat_x = alpha * x.detach() + (1 - alpha) * y.detach()

        loss_sum: Loss = Loss(torch.tensor(0))

        for loss in self.__losses:
            loss_sum += loss(Dx, Dy)

        for pen in self.__penalties:
            loss_sum += pen.__call__(lambda arr: self.forward(arr[0]), [hat_x])

        return loss_sum








