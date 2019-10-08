from typing import List

import torch
from torch import Tensor

from framework.Loss import Loss
from framework.gan.loss.penalties.DiscriminatorPenalty import DiscriminatorPenalty, GradientDiscriminatorPenalty, \
    ApproxGradientDiscriminatorPenalty


class LipschitzPenalty(GradientDiscriminatorPenalty):

    def __init__(self, weight: float):
        self.weight = weight

    def _compute(self, gradients: Tensor) -> Loss:

        gradients: Tensor = gradients.view((gradients.size(0), -1))
        gradient_penalty_value = ((gradients.norm(2, dim=1) - 1)**2).mean()
        return Loss(self.weight * gradient_penalty_value)


class ApproxLipschitzPenalty(ApproxGradientDiscriminatorPenalty):

    def __init__(self, weight: float):
        self.weight = weight

    def _compute(self, delta: Tensor) -> Loss:

        gradient_penalty_value = delta.relu().norm(2, dim=1).pow(2).mean()

        return Loss(self.weight * gradient_penalty_value)
