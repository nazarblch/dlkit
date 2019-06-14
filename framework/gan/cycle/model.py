from typing import Callable, Generic, TypeVar, Iterable
import torch
from torch import nn, Tensor, optim
from framework.Loss import Loss
from framework.segmentation.Mask import Mask
from itertools import chain

T1 = TypeVar('T1', Tensor, Mask, Iterable[Tensor])
T2 = TypeVar('T2', Tensor, Mask, Iterable[Tensor])


class CycleGAN(Generic[T1, T2]):

    def __init__(self,
                 g_forward: nn.Module,
                 g_backward: nn.Module,
                 loss_1: Callable[[T1, T1], Loss],
                 loss_2: Callable[[T2, T2], Loss],
                 lr: float = 0.0001,
                 betas=(0.5, 0.999)):

        self.g_forward = g_forward
        self.g_backward = g_backward
        self.loss_1 = loss_1
        self.loss_2 = loss_2

        self.opt = optim.Adam(
            chain(
              g_forward.parameters(),
              g_backward.parameters()
            ),
            lr=lr,
            betas=betas)

    def loss_forward(self, condition: T1) -> Loss:

        condition_pred: T1 = self.g_backward(self.g_forward(condition))

        return self.loss_1(condition_pred, condition)

    def loss_backward(self, condition: T2) -> Loss:
        condition_pred: T2 = self.g_forward(self.g_backward(condition))

        return self.loss_2(condition_pred, condition)

    def train(self, t1: Tensor, t2: Tensor):

        self.g_forward.zero_grad()
        self.g_backward.zero_grad()
        self.loss_forward(t1).minimize()
        self.opt.step()

        self.g_forward.zero_grad()
        self.g_backward.zero_grad()
        self.loss_backward(t2).minimize()
        self.opt.step()
