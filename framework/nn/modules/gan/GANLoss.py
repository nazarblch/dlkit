from abc import ABC, abstractmethod
from typing import List, Callable, Any, overload

import torch
from torch import Tensor

from framework.Loss import Loss
from framework.nn.modules.gan.DiscriminatorPenalty import DiscriminatorPenalty


class GANLoss(ABC):

    __penalties: List[DiscriminatorPenalty] = []

    def compute_discriminator_loss(self, discriminator: Callable[[List[Tensor]], Tensor],
                           x: List[Tensor],
                           y: List[Tensor]) -> Loss:
        x_detach = [xi.detach().requires_grad_(True) for xi in x]
        y_detach = [yi.detach().requires_grad_(True) for yi in y]

        Dx = discriminator(x_detach)
        Dy = discriminator(y_detach)

        loss_sum: Loss = self.discriminator_loss(Dx, Dy)

        for pen in self.get_penalties():
            loss_sum = loss_sum - pen.__call__(
                Dx / 2,
                x_detach
            )
            loss_sum = loss_sum - pen.__call__(
                Dy / 2,
                y_detach
            )

        return loss_sum

    @abstractmethod
    def discriminator_loss(self, dx: Tensor, dy: Tensor) -> Loss: pass

    @abstractmethod
    def generator_loss(self, dgz: Tensor) -> Loss: pass

    def add_penalty(self, pen: DiscriminatorPenalty):
        self.__penalties.append(pen)
        return self

    def add_penalties(self, pens: List[DiscriminatorPenalty]) -> None:
        self.__penalties.extend(pens)

    def get_penalties(self) -> List[DiscriminatorPenalty]:
        return self.__penalties

    def __add__(self, other):

        obj = GANLossObject(
            lambda dx, dy: self.discriminator_loss(dx, dy) + other.discriminator_loss(dx, dy),
            lambda dgz: self.generator_loss(dgz) + other.generator_loss(dgz)
        )

        obj.add_penalties(self.__penalties)
        obj.add_penalties(other.get_penalties())
        return obj

    def __mul__(self, weight: float):

        obj = GANLossObject(
            lambda dx, dy: self.discriminator_loss(dx, dy) * weight,
            lambda dgz: self.generator_loss(dgz) * weight
        )

        obj.add_penalties(self.__penalties)
        return obj


class GANLossObject(GANLoss):

    def __init__(self,
                 discriminator_loss: Callable[[Tensor, Tensor], Loss],
                 generator_loss: Callable[[Tensor], Loss]):
        self.d_loss = discriminator_loss
        self.g_loss = generator_loss

    def discriminator_loss(self, real: Tensor, fake: Tensor) -> Loss:
        return self.d_loss(real, fake)

    def generator_loss(self, fake: Tensor) -> Loss:
        return self.g_loss(fake)



