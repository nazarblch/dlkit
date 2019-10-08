from torch import Tensor

from framework.Loss import Loss
from framework.gan.GANLoss import GANLoss
from framework.gan.loss.penalties.lipschitz import LipschitzPenalty, ApproxLipschitzPenalty


class WassersteinLoss(GANLoss):

    def __init__(self, penalty_weight: float = 1):
        self.add_penalty(ApproxLipschitzPenalty(penalty_weight))

    def generator_loss(self, dgz: Tensor, real: Tensor, fake: Tensor) -> Loss:
        return Loss(-dgz.mean())

    def discriminator_loss(self, d_real: Tensor, d_fake: Tensor) -> Loss:
        discriminator_loss = (d_real).mean() - d_fake.mean()

        return Loss(discriminator_loss)
