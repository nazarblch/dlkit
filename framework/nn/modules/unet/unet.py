import torch
from torch import nn, Tensor
from typing import Callable, List


class UNet(nn.Module):

    def __init__(self,
                 n_down: int,
                 in_block: nn.Module,
                 out_block: nn.Module,
                 middle_block: nn.Module,
                 down_block: Callable[[int], nn.Module],
                 up_block: Callable[[int], nn.Module]
                 ):
        super(UNet, self).__init__()

        self.in_block: nn.Module = in_block
        self.out_block: nn.Module = out_block
        self.middle_block: nn.Module = middle_block

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

    def forward(self, *x: Tensor) -> Tensor:

        down = self._down(x[0])

        middle: Tensor = self.middle_block(down[-1])

        return self._up(middle, down)


