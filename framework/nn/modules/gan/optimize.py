from typing import Iterator

from torch import optim, Tensor

from framework.Loss import Loss


class GANOptimizer:

    def __init__(self,
                 generator_parameters: Iterator[Tensor],
                 discriminator_parameters: Iterator[Tensor],
                 learning_rate: float,
                 betas=(0.5, 0.99)):

        self.optD = optim.Adam(discriminator_parameters, lr=2*learning_rate, betas=betas)
        self.optG = optim.Adam(generator_parameters, lr=learning_rate, betas=betas)

    def train_step(self, generator_loss: Loss, discriminator_loss: Loss):

        self.optD.zero_grad()
        (-discriminator_loss).backward()
        self.optD.step()

        self.optG.zero_grad()
        generator_loss.backward()
        self.optG.step()


