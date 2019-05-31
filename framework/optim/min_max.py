from typing import Iterator

from torch import optim, Tensor

from framework.Loss import Loss


class MinMaxLoss:
    def __init__(self, min_loss: Loss, max_loss: Loss):
        self.min_loss = min_loss
        self.max_loss = max_loss

    def add_min_loss(self, loss: Loss):
        self.min_loss += loss

    def add_max_loss(self, loss: Loss):
        self.max_loss += loss


class MinMaxParameters:
    def __init__(self,
                 min_parameters: Iterator[Tensor],
                 max_parameters: Iterator[Tensor]):
        self.min_parameters = min_parameters
        self.max_parameters = max_parameters


class MinMaxOptimizer:

    def __init__(self,
                 parameters: MinMaxParameters,
                 min_learning_rate: float,
                 max_learning_rate: float,
                 betas=(0.5, 0.99)):

        self.opt_max = optim.Adam(parameters.max_parameters,
                                  lr=max_learning_rate,
                                  betas=betas)
        self.opt_min = optim.Adam(parameters.min_parameters,
                                  lr=min_learning_rate,
                                  betas=betas)

    def train_step(self, loss: MinMaxLoss):

        self.opt_max.zero_grad()
        loss.max_loss.maximize()
        self.opt_max.step()

        self.opt_min.zero_grad()
        loss.min_loss.minimize()
        self.opt_min.step()


