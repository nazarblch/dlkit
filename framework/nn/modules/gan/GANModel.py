from abc import ABC, abstractmethod

from torch import Tensor
import torch

from framework.Loss import Loss
from framework.nn.modules.gan.ConditionalDiscriminator import ConditionalDiscriminator
from framework.nn.modules.gan.ConditionalGenerator import ConditionalGenerator
from framework.nn.modules.gan.Discriminator import Discriminator
from framework.nn.modules.gan.GANLoss import GANLoss
from framework.nn.modules.gan.Generator import Generator
from framework.nn.modules.gan.loss_pair import GANLossPair
from framework.nn.modules.gan.optimize import GANParameters


class GANModel:

    def __init__(self, generator: Generator, discriminator: Discriminator, loss: GANLoss):
        self.generator = generator
        self.discriminator = discriminator
        self.loss = loss

    def discriminator_loss(self, real: Tensor, fake: Tensor) -> Loss:

        return self.loss.compute_discriminator_loss(
            lambda arr: self.discriminator.forward(arr[0]),
            [real],
            [fake]
        )

    def generator_loss(self, real: Tensor, fake: Tensor) -> Loss:
        DGz = self.discriminator.forward(fake)
        return self.loss.generator_loss(DGz, real, fake)

    def loss_pair(self, real: Tensor) -> GANLossPair:
        fake = self.generator.forward(real.size(0))
        return GANLossPair(
            self.generator_loss(real, fake),
            self.discriminator_loss(real, fake)
        )

    def parameters(self) -> GANParameters:
        return GANParameters(self.generator.parameters(), self.discriminator.parameters())


class ConditionalGANModel:

    def __init__(self, generator: ConditionalGenerator, discriminator: ConditionalDiscriminator, loss: GANLoss):
        self.generator = generator
        self.discriminator = discriminator
        self.loss = loss

    def discriminator_loss(self, real: Tensor, fake: Tensor, condition: Tensor) -> Loss:

        return self.loss.compute_discriminator_loss(
            lambda arr: self.discriminator.forward(arr[0], arr[1]),
            [real, condition],
            [fake, condition]
        )

    def generator_loss(self, real: Tensor, fake: Tensor, condition: Tensor) -> Loss:
        DGz = self.discriminator.forward(fake, condition)
        return self.loss.generator_loss(DGz, real, fake)

    def loss_pair(self, real: Tensor, condition: Tensor) -> GANLossPair:
        fake = self.generator.forward(real.size(0), condition)
        return GANLossPair(
            self.generator_loss(real, fake, condition),
            self.discriminator_loss(real, fake, condition)
        )

    def parameters(self) -> GANParameters:
        return GANParameters(self.generator.parameters(), self.discriminator.parameters())
