from torch import Tensor
import numpy as np
from framework.Loss import LossModule, Loss
from gan.gan_model import ConditionalGANModel
from framework.segmentation.loss.base import SegmentationLoss


class InPaintLoss(SegmentationLoss):

    def __init__(self,
                 gan: ConditionalGANModel,
                 weight: float = 1.0):
        self.weight = weight
        self.gan = gan

    def forward(self, image: Tensor, sp: Tensor, segm: Tensor) -> Loss:

        painted = self.gan.generator(segm)
        generator_loss = self.gan.generator_loss(image, painted, segm)

        return generator_loss

