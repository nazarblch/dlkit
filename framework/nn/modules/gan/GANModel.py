from abc import ABC, abstractmethod

from torch import Tensor

from framework.Loss import Loss
from framework.nn.modules.gan.ConditionalDiscriminator import ConditionalDiscriminator
from framework.nn.modules.gan.ConditionalGenerator import ConditionalGenerator
from framework.nn.modules.gan.Discriminator import Discriminator
from framework.nn.modules.gan.GANLoss import GANLoss
from framework.nn.modules.gan.Generator import Generator


class GANModel:

    def __init__(self, generator: Generator, discriminator: Discriminator, loss: GANLoss):
        self.generator = generator
        self.discriminator = discriminator
        self.loss = loss

    def discriminator_loss(self, real: Tensor, fake: Tensor) -> Loss:
        Dreal = self.discriminator.forward(real.detach())
        Dfake = self.discriminator.forward(fake.detach())

        loss_sum: Loss = self.loss.discriminator_loss(Dreal, Dfake)

        for pen in self.loss.get_penalties():
            loss_sum += pen.__call__(Dreal, [real.detach()])
            loss_sum += pen.__call__(Dfake, [fake.detach()])

        return loss_sum

    def generator_loss(self, fake: Tensor) -> Loss:
        DGz = self.discriminator.forward(fake)
        return self.loss.generator_loss(DGz)


class ConditionalGANModel:

    def __init__(self, generator: ConditionalGenerator, discriminator: ConditionalDiscriminator, loss: GANLoss):
        self.generator = generator
        self.discriminator = discriminator
        self.loss = loss

    def discriminator_loss(self, real: Tensor, fake: Tensor, condition: Tensor) -> Loss:
        Dreal = self.discriminator.forward(real.detach(), condition.detach())
        Dfake = self.discriminator.forward(fake.detach(), condition.detach())

        loss_sum: Loss = self.loss.discriminator_loss(Dreal, Dfake)

        for pen in self.loss.get_penalties():
            loss_sum += pen.__call__(Dreal, [real.detach(), condition.detach()])
            loss_sum += pen.__call__(Dfake, [fake.detach(), condition.detach()])

        return loss_sum

    def generator_loss(self, fake: Tensor, condition: Tensor) -> Loss:
        DGz = self.discriminator.forward(fake, condition)
        return self.loss.generator_loss(DGz)
