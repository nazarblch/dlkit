from torch import nn, Tensor
import torch
from typing import Callable, Optional, List
import math


class LocalPairwiseMap2D:

    def __init__(self, kernel_size: int, stride: int = 1):
        self.kernel_size = kernel_size
        self.stride = stride

    """
    f: di, dj, T_center, Tij => T[n * n_block, nc*, 1]
    """
    def __call__(self, tensor: Tensor, f: Callable[[int, int, Tensor, Tensor], Tensor]) -> Tensor:
        n = tensor.size(0)
        nc = tensor.size(1)

        fold: Tensor = nn.functional.unfold(tensor, self.kernel_size, stride=self.stride)
        n_block = fold.shape[-1]
        fold = fold.transpose(1, 2).reshape(n, n_block, nc, self.kernel_size, self.kernel_size)

        i_center = int(self.kernel_size // 2)
        j_center = int(self.kernel_size // 2)
        t_center = fold[:, :, :, i_center, j_center]

        res: List[Tensor] = []

        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                if i == i_center and j == j_center:
                    continue

                t_ij = fold[:, :, :, i, j]
                di = int(math.fabs(i - i_center))
                dj = int(math.fabs(j - j_center))
                res.append(
                    f(di, dj, t_center, t_ij).view(n, n_block, -1, 1)
                )

        return torch.cat(res, dim=3)

