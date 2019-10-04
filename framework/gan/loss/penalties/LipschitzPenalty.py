from typing import List

import torch
from torch import Tensor

from framework.Loss import Loss
from framework.gan.DiscriminatorPenalty import DiscriminatorPenalty


class LipschitzPenalty(DiscriminatorPenalty):

    def __init__(self, weight: float):
        self.weight = weight

    def __call__(self, dx: Tensor, x: List[Tensor]) -> Loss:

        gradients = torch.autograd.grad(outputs=dx,
                                        inputs=x,
                                        grad_outputs=torch.ones(dx.size(), device=dx.device),
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)[0]

        gradients: Tensor = gradients.view(gradients.size(0), -1)
        gradient_penalty_value = ((gradients.norm(2, dim=1) - 1)**2).mean()
        return Loss(self.weight * gradient_penalty_value)
