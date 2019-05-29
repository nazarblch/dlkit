
import torch
from torch import Tensor, nn

from framework.Loss import Loss
from framework.nn.modules.gan.GANLoss import GANLoss
from framework.nn.modules.gan.penalties.LipschitzPenalty import LipschitzPenalty


class WassersteinLoss(GANLoss):

    def __init__(self, penalty_weight: float = 1):
        self.add_penalty(LipschitzPenalty(penalty_weight))

    def generator_loss(self, dgz: Tensor, real: Tensor, fake: Tensor) -> Loss:
        return Loss(-dgz.mean())

    def discriminator_loss(self, d_real: Tensor, d_fake: Tensor) -> Loss:
        discriminator_loss = d_real.mean() - d_fake.mean()

        return Loss(discriminator_loss)
