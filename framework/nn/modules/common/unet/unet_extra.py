import torch
from torch import nn, Tensor
from typing import Callable, Optional, List, overload

from framework.nn.modules.common.unet.unet import UNet


class UNetExtra(UNet):

    def __init__(self,
                 n_down: int,
                 in_block: nn.Module,
                 out_block: nn.Module,
                 middle_block: nn.Module,
                 middle_block_extra: nn.Module,
                 down_block: Callable[[int], nn.Module],
                 up_block: Callable[[int], nn.Module]
                 ):
        super(UNetExtra, self).__init__(n_down, in_block, out_block, middle_block, down_block, up_block)

        self.middle_block_extra: nn.Module = middle_block_extra

    def forward(self, x: Tensor, x_extra: Tensor) -> Tensor:

        down = self._down(x)

        middle: Tensor = self.middle_block(down[-1])

        middle = self.middle_block_extra(
            torch.cat([middle, x_extra], dim=1)
        )

        return self._up(middle, down)


