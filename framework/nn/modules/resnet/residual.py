from torch import nn, Tensor
from enum import Enum

from typing import Dict, List


class PaddingType(Enum):
    REFLECT = 'reflect'
    REPLICATE = 'replicate'
    NONE = 'zero'


class ResidualBlock(nn.Module):

    def __init__(self,
                 dim: int,
                 padding_type: PaddingType,
                 norm_layer: nn.Module,
                 activation=nn.ReLU(True),
                 use_dropout=False):
        super(ResidualBlock, self).__init__()

        self.padding2module: Dict[PaddingType, List] = {
            PaddingType.REFLECT: [nn.ReflectionPad2d(1)],
            PaddingType.REPLICATE: [nn.ReplicationPad2d(1)],
            PaddingType.NONE: []
        }

        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim: int,
                         padding_type: PaddingType,
                         norm_layer: nn.Module,
                         activation: nn.Module,
                         use_dropout: bool):

        conv_block = []
        p = 0 if padding_type != PaddingType.NONE else 1
        padding_module = self.padding2module[padding_type]

        conv_block += padding_module

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        conv_block += padding_module

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x: Tensor):
        out = x + self.conv_block(x)
        return out


class ResidualNet(nn.Module):

    def __init__(self,
                 dim: int,
                 n_blocks: int,
                 norm_layer: nn.Module = nn.BatchNorm2d,
                 padding_type: PaddingType = PaddingType.REFLECT):
        super(ResidualNet, self).__init__()

        model_list = []
        for i in range(n_blocks):
            model_list += [ResidualBlock(dim, padding_type=padding_type, norm_layer=norm_layer)]

        self.model = nn.Sequential(*model_list)

    def forward(self, x: Tensor):
        return self.model(x)
