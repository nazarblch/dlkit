from typing import Iterator

from torch import optim, Tensor

from framework.Loss import Loss
from framework.nn.modules.gan.GANModel import GANLossPair


class GANParameters:
    def __init__(self,
                 generator_parameters: Iterator[Tensor],
                 discriminator_parameters: Iterator[Tensor]):
        self.generator_parameters = generator_parameters
        self.discriminator_parameters = discriminator_parameters


class GANOptimizer:

    def __init__(self,
                 parameters: GANParameters,
                 learning_rate: float,
                 betas=(0.5, 0.9)):

        self.optD = optim.Adam(parameters.discriminator_parameters, lr=3*learning_rate, betas=betas)
        self.optG = optim.Adam(parameters.generator_parameters, lr=learning_rate, betas=betas)

    def train_step(self, loss: GANLossPair):

        self.optD.zero_grad()
        loss.discriminator_loss.maximize()
        self.optD.step()

        self.optG.zero_grad()
        loss.generator_loss.minimize()
        self.optG.step()


