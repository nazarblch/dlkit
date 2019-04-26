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
        real_detach = real.detach()
        fake_detach = fake.detach()
        Dreal = self.discriminator.forward(real_detach)
        Dfake = self.discriminator.forward(fake_detach)

        loss_sum: Loss = self.loss.discriminator_loss(Dreal, Dfake)

        for pen in self.loss.get_penalties():
            real_detach.requires_grad = True
            fake_detach.requires_grad = True
            loss_sum -= pen.__call__(
                self.discriminator.forward(real_detach) / 2,
                [real_detach]
            )
            loss_sum -= pen.__call__(
                self.discriminator.forward(fake_detach) / 2,
                [fake_detach]
            )

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
        real_detach = real.detach()
        fake_detach = fake.detach()
        condition_detach = condition.detach()
        Dreal = self.discriminator.forward(real_detach, condition_detach)
        Dfake = self.discriminator.forward(fake_detach, condition_detach)

        loss_sum: Loss = self.loss.discriminator_loss(Dreal, Dfake)

        for pen in self.loss.get_penalties():
            real_detach.requires_grad = True
            fake_detach.requires_grad = True
            condition_detach.requires_grad = True
            loss_sum -= pen.__call__(self.discriminator.forward(real_detach, condition_detach) / 2,
                                     [real_detach, condition_detach])
            loss_sum -= pen.__call__( self.discriminator.forward(fake_detach, condition_detach) / 2, [fake_detach, condition_detach])

        return loss_sum

    def generator_loss(self, fake: Tensor, condition: Tensor) -> Loss:
        DGz = self.discriminator.forward(fake, condition)
        return self.loss.generator_loss(DGz)
