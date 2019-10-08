import torch

from torch import Tensor, nn
from framework.Loss import Loss
from framework.gan.GANLoss import GANLoss
from framework.gan.conditional import ConditionalGenerator, ConditionalDiscriminator
from framework.gan.discriminator import Discriminator
from framework.gan.generator import Generator
from framework.optim.min_max import MinMaxParameters, MinMaxLoss


class GANModel:

    def __init__(self, generator: nn.Module, discriminator: nn.Module, loss: GANLoss):
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

    def loss_pair(self, real: Tensor, noize: Tensor) -> MinMaxLoss:

        fake = self.generator.forward(noize)
        return MinMaxLoss(
            self.generator_loss(real, fake),
            self.discriminator_loss(real, fake)
        )

    def parameters(self) -> MinMaxParameters:
        return MinMaxParameters(self.generator.parameters(), self.discriminator.parameters())


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

    def loss_pair(self, real: Tensor, condition: Tensor, *additional_input: Tensor) -> MinMaxLoss:
        fake = self.generator.forward(condition, *additional_input)
        return MinMaxLoss(
            self.generator_loss(real, fake, condition),
            self.discriminator_loss(real, fake, condition)
        )

    def parameters(self) -> MinMaxParameters:
        return MinMaxParameters(self.generator.parameters(), self.discriminator.parameters())
