import torch
from abc import ABC, abstractmethod
from typing import Callable, List
from torch import Tensor
import numpy as np
from framework.Loss import Loss


class DiscriminatorPenalty(ABC):

    @abstractmethod
    def __call__(
            self,
            discriminator: Callable[[List[Tensor]], Tensor],
            dx: Tensor,
            dy: Tensor,
            x: List[Tensor],
            y: List[Tensor]) -> Loss: pass


class GradientDiscriminatorPenalty(DiscriminatorPenalty):

    @abstractmethod
    def _compute(self, grad: Tensor) -> Loss: pass

    def gradient_point(self, x: List[Tensor], y: List[Tensor]) -> List[Tensor]:
        eps = np.random.random_sample()
        x0: List[Tensor] = [(xi * eps + yi * (1 - eps)).detach().requires_grad_(True) for xi, yi in
                            zip(x, y)]
        return x0

    def __call__(
            self,
            discriminator: Callable[[List[Tensor]], Tensor],
            dx: Tensor,
            dy: Tensor,
            x: List[Tensor],
            y: List[Tensor]) -> Loss:

        x0 = self.gradient_point(x, y)
        dx0: Tensor = discriminator(x0)

        grads: List[Tensor] = torch.autograd.grad(outputs=dx0,
                                   inputs=x0,
                                   grad_outputs=torch.ones(dx0.shape, device=dx0.device),
                                   create_graph=True,
                                   retain_graph=True,
                                   only_inputs=True)

        if len(grads) == 1:
            grad = grads[0]
        else:
            grad = torch.cat(*grads, dim=1)

        return self._compute(grad)


class ApproxGradientDiscriminatorPenalty(DiscriminatorPenalty):

    @abstractmethod
    def _compute(self, delta: Tensor) -> Loss: pass

    def __call__(
            self,
            discriminator: Callable[[List[Tensor]], Tensor],
            dx: Tensor,
            dy: Tensor,
            x: List[Tensor],
            y: List[Tensor]) -> Loss:

        n = x[0].shape[0]

        if len(x) > 1:
            x = torch.cat(*x, dim=1)
            y = torch.cat(*y, dim=1)
        else:
            x = x[0]
            y = y[0]

        norm = (x - y).view(n, -1).norm(2, dim=1).detach().view(n, 1)
        delta = (dx - dy).abs() - norm

        return self._compute(delta)

