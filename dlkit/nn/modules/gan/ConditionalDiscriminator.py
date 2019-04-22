from abc import ABC, abstractmethod

from typing import Callable, List, Tuple

import torch
from torch import Tensor

from dlkit.Loss import Loss
from dlkit.nn.modules.gan.DiscriminatorPenalty import DiscriminatorPenalty


class ConditionalDiscriminator(torch.nn.Module, ABC):

    __penalties: List[DiscriminatorPenalty]
    __losses: List[Callable[[Tensor, Tensor], Loss]]

    def add_penalty(self, pen: DiscriminatorPenalty) -> None:
        self.__penalties.append(pen)

    def add_pair_loss(self, loss: Callable[[Tensor, Tensor], Loss]) -> None:
        self.__losses.append(loss)

    @abstractmethod
    def forward(self, x: Tensor, condition: Tensor) -> Tensor: pass

    def loss(self, x: Tensor, y: Tensor, condition: Tensor) -> Loss:
        Dx = self.forward(x.detach(), condition.detach())
        Dy = self.forward(y.detach(), condition.detach())

        alpha = torch.rand(x.size(0), *((1,) * (x.ndimension() - 1)), device=x.device)
        hat_x: Tensor = alpha * x.detach() + (1 - alpha) * y.detach()

        loss_sum: Loss = Loss(torch.tensor(0))

        for loss in self.__losses:
            loss_sum += loss(Dx, Dy)

        for pen in self.__penalties:
            f = lambda x_and_condition: self.forward(x_and_condition[0], x_and_condition[1])
            loss_sum += pen.__call__(f, [hat_x, condition.detach()])

        return loss_sum
