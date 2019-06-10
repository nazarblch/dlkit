from abc import ABC, abstractmethod
from typing import List, Callable

import numpy
from torch import Tensor

from framework.Loss import Loss
from framework.gan.DiscriminatorPenalty import DiscriminatorPenalty


class GANLoss(ABC):

    __penalties: List[DiscriminatorPenalty] = []

    def compute_discriminator_loss(self,
                                   discriminator: Callable[[List[Tensor]], Tensor],
                                   x: List[Tensor],
                                   y: List[Tensor]) -> Loss:
        x_detach = [xi.detach() for xi in x]
        y_detach = [yi.detach() for yi in y]

        Dx = discriminator(x_detach)
        Dy = discriminator(y_detach)

        loss_sum: Loss = self.discriminator_loss(Dx, Dy)

        rand = numpy.random

        for pen in self.get_penalties():
            eps = rand.random_sample()
            x0: List[Tensor] = [(xi * eps + yi * (1 - eps)).detach().requires_grad_(True) for xi, yi in zip(x_detach, y_detach)]
            D0 = discriminator(x0)
            loss_sum = loss_sum - pen.__call__(D0, x0)

        return loss_sum

    @abstractmethod
    def discriminator_loss(self, dx: Tensor, dy: Tensor) -> Loss: pass

    @abstractmethod
    def generator_loss(self, dgz: Tensor, real: Tensor, fake: Tensor) -> Loss: pass

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
            lambda dgz, real, fake: self.generator_loss(dgz, real, fake) + other.generator_loss(dgz, real, fake)
        )

        obj.add_penalties(self.__penalties)
        obj.add_penalties(other.get_penalties())
        return obj

    def __mul__(self, weight: float):

        obj = GANLossObject(
            lambda dx, dy: self.discriminator_loss(dx, dy) * weight,
            lambda dgz, real, fake: self.generator_loss(dgz, real, fake) * weight
        )

        obj.add_penalties(self.__penalties)

        return obj


class GANLossObject(GANLoss):

    def __init__(self,
                 discriminator_loss: Callable[[Tensor, Tensor], Loss],
                 generator_loss: Callable[[Tensor, Tensor, Tensor], Loss]):
        self.d_loss = discriminator_loss
        self.g_loss = generator_loss

    def discriminator_loss(self, dx: Tensor, dy: Tensor) -> Loss:
        return self.d_loss(dx, dy)

    def generator_loss(self, dgz: Tensor, real: Tensor, fake: Tensor) -> Loss:
        return self.g_loss(dgz, real, fake)


