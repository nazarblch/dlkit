from abc import ABC, abstractmethod
from typing import Callable, List

from torch import Tensor, nn
import numpy as np
import torch

from framework.Loss import Loss, LossModule
from framework.parallel import ParallelConfig


class PenalizedSegmentation:

    def __init__(self, segmentation: nn.Module):
        self.segmentation = nn.DataParallel(
             segmentation.to(ParallelConfig.MAIN_DEVICE),
             ParallelConfig.GPU_IDS
        )
        self.penalties: List[LossModule] = []
        self.opt = torch.optim.SGD(self.segmentation.parameters(), lr=0.1, momentum=0.9)

    def forward(self, image: Tensor) -> Tensor:
        return self.segmentation(image)

    def __call__(self, image: Tensor) -> Tensor:
        return self.forward(image)

    def add_penalty(self, pen: LossModule):
        self.penalties.append(pen)

    def train(self, image: Tensor):
        segm = self.segmentation(image)
        loss: Loss = Loss.ZERO()
        for pen in self.penalties:
            loss = loss + pen.forward(image, segm)
        loss.minimize_step(self.opt)


