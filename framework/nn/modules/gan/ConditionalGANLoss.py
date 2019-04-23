from abc import ABC, abstractmethod
from typing import List, Callable

import torch
from torch import Tensor

from dlkit.Loss import Loss
from dlkit.nn.modules.gan.ConditionalDiscriminator import ConditionalDiscriminator
from dlkit.nn.modules.gan.DiscriminatorPenalty import DiscriminatorPenalty


class ConditionalGANLoss(ABC):

    __penalties: List[DiscriminatorPenalty]
    __disc_losses: List[Callable[[Tensor, Tensor], Loss]]
    __gen_losses: List[Callable[[Tensor], Loss]]

    def __init__(self, discriminator: ConditionalDiscriminator):
        self.discriminator = discriminator

    @abstractmethod
    def _discriminator_loss(self, x: Tensor, y: Tensor) -> Loss: pass

    @abstractmethod
    def _generator_loss(self, x: Tensor) -> Loss: pass

    def add_penalty(self, pen: DiscriminatorPenalty) -> None:
        self.__penalties.append(pen)

    def add_discriminator_loss(self, loss: Callable[[Tensor, Tensor], Loss]) -> None:
        self.__disc_losses.append(loss)

    def add_generator_loss(self, loss: Callable[[Tensor], Loss]) -> None:
        self.__gen_losses.append(loss)

    def discriminator_loss(self, x: Tensor, y: Tensor, condition: Tensor) -> Loss:
        Dx = self.discriminator.forward(x.detach(), condition.detach())
        Dy = self.discriminator.forward(y.detach(), condition.detach())

        alpha = torch.rand(x.size(0), *((1,) * (x.ndimension() - 1)), device=x.device)
        hat_x = alpha * x.detach() + (1 - alpha) * y.detach()

        loss_sum: Loss = Loss(self._discriminator_loss(Dx, Dy))

        for loss in self.__disc_losses:
            loss_sum += loss(Dx, Dy)

        for pen in self.__penalties:
            f = lambda x_and_condition: self.discriminator.forward(x_and_condition[0], x_and_condition[1])
            loss_sum += pen.__call__(f, [hat_x, condition.detach()])

        return loss_sum

    def generator_loss(self, Gz: Tensor) -> Loss:

        loss_sum: Loss = Loss(self._generator_loss(Gz))

        for loss in self.__gen_losses:
            loss_sum += loss(Gz)

        return loss_sum

