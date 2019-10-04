import torch
from torch import Tensor, nn

from framework.Loss import Loss
from framework.gan.GANLoss import GANLoss


class HingeLoss(GANLoss):

    def generator_loss(self, dgz: Tensor, real: Tensor, fake: Tensor) -> Loss:
        return Loss(-dgz.mean())

    def discriminator_loss(self, d_real: Tensor, d_fake: Tensor) -> Loss:
        discriminator_loss = (1 - d_real).relu().mean() + (1 + d_fake).relu().mean()
        return Loss(-discriminator_loss)
