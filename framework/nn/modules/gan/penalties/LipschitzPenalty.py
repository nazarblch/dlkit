from typing import Callable, List

import torch
from bokeh.core.property import override
from torch import Tensor

from framework.Loss import Loss
from framework.nn.modules.gan.DiscriminatorPenalty import DiscriminatorPenalty


class LipschitzPenalty(DiscriminatorPenalty):

    def __init__(self, weight: float):
        self.weight = weight

    def __call__(self, discriminator: Callable[[List[Tensor]], Tensor], x: List[Tensor]) -> Loss:

        for xi in x:
            xi.requires_grad_(True)

        Dx: Tensor = discriminator(x)

        gradients = torch.autograd.grad(outputs=Dx,
                                        inputs=x,
                                        grad_outputs=torch.ones(Dx.size(), device=Dx.device),
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)[0]

        gradients: Tensor = gradients.view(gradients.size(0), -1)
        gradient_penalty_value = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return self.weight * gradient_penalty_value