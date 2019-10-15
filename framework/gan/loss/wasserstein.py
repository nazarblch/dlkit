from torch import Tensor

from framework.Loss import Loss
from framework.gan.discriminator import Discriminator
from framework.gan.loss.gan_loss import GANLoss
from framework.gan.loss.penalties.lipschitz import ApproxLipschitzPenalty


class WassersteinLoss(GANLoss):

    def __init__(self, discriminator: Discriminator, penalty_weight: float = 1):
        super().__init__(discriminator)
        self.add_penalty(ApproxLipschitzPenalty(penalty_weight))

    def _generator_loss(self, dgz: Tensor, real: Tensor, fake: Tensor) -> Loss:
        return Loss(-dgz.mean())

    def _discriminator_loss(self, d_real: Tensor, d_fake: Tensor) -> Loss:
        discriminator_loss = (d_real).mean() - d_fake.mean()

        return Loss(discriminator_loss)
