import torch
from torch import nn, Tensor
from typing import Callable, Optional, List


class UNet(nn.Module):

    def __init__(self,
                 n_down: int,
                 in_block: nn.Module,
                 out_block: nn.Module,
                 middle_block_1: nn.Module,
                 middle_block_2: Optional[nn.Module],
                 down_block: Callable[[int], nn.Module],
                 up_block: Callable[[int], nn.Module]
                 ):
        super(UNet, self).__init__()

        self.in_block: nn.Module = in_block
        self.out_block: nn.Module = out_block
        self.middle_block_1: nn.Module = middle_block_1
        self.middle_block_2: nn.Module = middle_block_2

        self.down_blocks = nn.ModuleList([
            down_block(i) for i in range(n_down)
        ])

        self.up_blocks = nn.ModuleList([
            up_block(i) for i in range(n_down)
        ])

    def _down(self, x: Tensor) -> List[Tensor]:

        down = [self.in_block.forward(x)]

        for layer in self.down_blocks:
            down.append(
                layer(down[-1])
            )

        return down

    def _up(self, middle: Tensor, down: List[Tensor]) -> Tensor:

        for i, layer in enumerate(self.up_blocks):
            middle = layer(
                torch.cat([middle, down[-1 - i]], dim=1)
            )

        return self.out_block(middle)

    def forward(self, x: Tensor, middle_x: Optional[Tensor]) -> Tensor:

        down = self._down(x)

        middle: Tensor = self.middle_block_1(down[-1])

        if middle_x is not None:
            middle = self.middle_block_2(
                torch.cat([middle, middle_x], dim=1)
            )

        return self._up(middle, down)


