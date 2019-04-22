from abc import ABC, abstractmethod

import torch
from torch import Tensor


class Generator(torch.nn.Module, ABC):

    def __init__(self, batch_size: int, noise_size: int, device: torch.device):
        self.noise_size: int = noise_size
        self.batch_size: int = batch_size
        self.device = device

    @abstractmethod
    def forward(self, noise: Tensor) -> Tensor: pass

    def forward(self) -> Tensor:

        z: Tensor = torch.FloatTensor([self.batch_size, self.noise_size],
                                      device=self.device
                                      ).normal_()

        return self.forward(z)
