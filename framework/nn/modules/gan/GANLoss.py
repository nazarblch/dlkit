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
    def _discriminator_loss(self, real: Tensor, fake: Tensor) -> Loss: pass

    @abstractmethod
    def _generator_loss(self, fake: Tensor) -> Loss: pass

    def add_penalty(self, pen: DiscriminatorPenalty) -> None:
        self.__penalties.append(pen)

    def add_discriminator_loss(self, loss: Callable[[Tensor, Tensor], Loss]) -> None:
        self.__disc_losses.append(loss)

    def add_generator_loss(self, loss: Callable[[Tensor], Loss]) -> None:
        self.__gen_losses.append(loss)

    def discriminator_loss(self, real: Tensor, fake: Tensor) -> Loss:
        Dx = self.discriminator.forward(real.detach())
        Dy = self.discriminator.forward(fake.detach())

        alpha = torch.rand(real.size(0), *((1,) * (real.ndimension() - 1)), device=real.device)
        hat_x = alpha * real.detach() + (1 - alpha) * fake.detach()

        loss_sum: Loss = self._discriminator_loss(Dx, Dy)

        for loss in self.__disc_losses:
            loss_sum += loss(Dx, Dy)

        for pen in self.__penalties:
            loss_sum += pen.__call__(lambda arr: self.discriminator.forward(arr[0]), [hat_x])

        return loss_sum

    def generator_loss(self, fake: Tensor) -> Loss:

        loss_sum: Loss = self._generator_loss(fake)

        for loss in self.__gen_losses:
            loss_sum += loss(fake)

        return loss_sum

    def __add__(self, other):
        return GANLossObject(
            self.discriminator,
            lambda real, fake: self.discriminator_loss(real, fake) + other.discriminator_loss(real, fake),
            lambda fake: self.generator_loss(fake) + other.generator_loss(fake)
        )

    def __mul__(self, weight: float):

        return GANLossObject(
            self.discriminator,
            lambda real, fake: self.discriminator_loss(real, fake) * weight,
            lambda fake: self.generator_loss(fake) * weight
        )


class GANLossObject(GANLoss):
    def _discriminator_loss(self, real: Tensor, fake: Tensor) -> Loss:
        pass

    def _generator_loss(self, fake: Tensor) -> Loss:
        pass

    def __init__(self, discriminator: Discriminator,
                 discriminator_loss: Callable[[Tensor, Tensor], Loss],
                 generator_loss: Callable[[Tensor], Loss]):
        super().__init__(discriminator)
        self.d_loss = discriminator_loss
        self.g_loss = generator_loss

    def discriminator_loss(self, real: Tensor, fake: Tensor) -> Loss:
        return self.d_loss(real, fake)

    def generator_loss(self, fake: Tensor) -> Loss:
        return self.g_loss(fake)

