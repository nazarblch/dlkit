from typing import Callable, List

import torch
from torch import Tensor

from framework.Loss import Loss
from framework.nn.modules.gan.DiscriminatorPenalty import DiscriminatorPenalty


class LipschitzPenalty(DiscriminatorPenalty):

    def __init__(self, weight: float):
        self.weight = weight

    def __call__(self, Dx: Tensor, x: List[Tensor]) -> Loss:

        gradients = torch.autograd.grad(outputs=Dx,
                                        inputs=x,
                                        grad_outputs=torch.ones(Dx.size(), device=Dx.device),
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)[0]

        gradients: Tensor = gradients.view(gradients.size(0), -1)
        gradient_penalty_value = torch.relu(gradients.norm(2, dim=1) - 1).mean()
        return Loss(self.weight * gradient_penalty_value)
