import torch
from torch import Tensor

from framework.nn.modules.gan.noise.Noise import Noise


class NormalNoise(Noise):

    def __init__(self, size: int, device: torch.device, mean: float = 0, std: float = 1):
        self.size: int = size
        self.device = device
        self.m = mean
        self.sd = std

    def sample(self, n: int) -> Tensor:
        return torch.FloatTensor(n, self.size).normal_(self.m, self.sd).to(self.device)

    def size(self) -> int:
        return self.size

