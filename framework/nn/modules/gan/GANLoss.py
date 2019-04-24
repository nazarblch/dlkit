from abc import ABC, abstractmethod
from typing import List, Callable

import torch
from torch import Tensor

from framework.Loss import Loss
from framework.nn.modules.gan.Discriminator import Discriminator
from framework.nn.modules.gan.DiscriminatorPenalty import DiscriminatorPenalty


class GANLoss(ABC):

    __penalties: List[DiscriminatorPenalty] = []
    __disc_losses: List[Callable[[Tensor, Tensor], Loss]] = []
    __gen_losses: List[Callable[[Tensor], Loss]] = []

    def __init__(self, discriminator: Discriminator):
        self.discriminator = discriminator

    @abstractmethod
    def _discriminator_loss(self, dx: Tensor, dy: Tensor) -> Loss: pass

    @abstractmethod
    def _generator_loss(self, gz: Tensor) -> Loss: pass

    def add_penalty(self, pen: DiscriminatorPenalty) -> None:
        self.__penalties.append(pen)

    def add_discriminator_loss(self, loss: Callable[[Tensor, Tensor], Loss]) -> None:
        self.__disc_losses.append(loss)

    def add_generator_loss(self, loss: Callable[[Tensor], Loss]) -> None:
        self.__gen_losses.append(loss)

    def discriminator_loss(self, x: Tensor, y: Tensor) -> Loss:
        Dx = self.discriminator.forward(x.detach())
        Dy = self.discriminator.forward(y.detach())

        alpha = torch.rand(x.size(0), *((1,) * (x.ndimension() - 1)), device=x.device)
        hat_x = alpha * x.detach() + (1 - alpha) * y.detach()

        loss_sum: Loss = self._discriminator_loss(Dx, Dy)

        for loss in self.__disc_losses:
            loss_sum += loss(Dx, Dy)

        for pen in self.__penalties:
            loss_sum += pen.__call__(lambda arr: self.discriminator.forward(arr[0]), [hat_x])

        return loss_sum

    def generator_loss(self, gz: Tensor) -> Loss:

        loss_sum: Loss = self._generator_loss(gz)

        for loss in self.__gen_losses:
            loss_sum += loss(gz)

        return loss_sum

