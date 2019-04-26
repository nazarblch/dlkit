from typing import Union, Tuple

import torch

from torch import nn, Tensor
import torch.nn.functional as F


class Upsample(nn.Module):

    def __init__(self,
                 size: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]]=None,
                 scale_factor=Union[float, Tuple[float]],
                 mode='bilinear'):
        super(Upsample, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, inputs: Tensor) -> Tensor:
        return F.interpolate(inputs, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                             align_corners=False)

    def extra_repr(self) -> str:
        params = []
        if self.size is not None:
            params.append('size={}'.format(self.size))
        if self.scale_factor is not None:
            params.append('scale_factor={}'.format(self.scale_factor))
        params.append('mode={}'.format(self.mode))
        return ', '.join(params)
