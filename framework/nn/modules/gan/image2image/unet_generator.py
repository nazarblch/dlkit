from functools import reduce
from typing import List

import torch
from torch import nn, Tensor

from framework.nn.modules.common.Down2xConv2d import Down2xConv2d
from framework.nn.modules.common.Up2xConv2d import Up4xConv2d, Up2xConv2d
from framework.nn.modules.common.View import View
from framework.nn.modules.common.unet.unet_extra import UNetExtra
from framework.nn.modules.gan.ConditionalGenerator import ConditionalGenerator as CG
from framework.nn.modules.gan.noise.Noise import Noise


class UNetGenerator(CG):

    def __init__(self, noise: Noise, image_size, in_channels, out_channels, gen_size=64, down_conv_count=5):
        super(UNetGenerator, self).__init__(noise)

        nc_max = 512

        middle_data_size = int(image_size * 2**(-down_conv_count))
        middle_nc = min(
            int(gen_size * 2 ** down_conv_count),
            nc_max
        )

        assert (middle_data_size >= 4)
        assert (middle_data_size % 4 == 0)

        def down_block_factory(index: int) -> nn.Module:
            mult = 2 ** index
            in_size = min(int(gen_size * mult), nc_max)
            out_size = min(2 * in_size, nc_max)
            return Down2xConv2d(in_size, out_size)

        def up_block_factory(index: int) -> nn.Module:
            mult = 2 ** (down_conv_count - index)
            in_size = min(int(gen_size * mult), nc_max)
            out_size = min(3 * in_size if index > 0 else 2 * in_size, 2 * nc_max)
            return Up2xConv2d(out_size, in_size)

        self.down_last_to_noise = nn.Sequential(
            View(-1, middle_nc * middle_data_size**2),
            nn.Linear(middle_nc * middle_data_size**2, noise.size()),
            nn.Tanh()
        )

        self.noise_up_modules = nn.Sequential(
            nn.Linear(2 * noise.size(), 2 * noise.size()),
            nn.ReLU(True),
            nn.Linear(2 * noise.size(), 2 * noise.size()),
            nn.ReLU(True),
            View(-1, 2 * noise.size(), 1, 1),
            Up4xConv2d(2 * noise.size(), middle_nc)
        )

        up_size = 4

        while up_size < middle_data_size:
            up_size *= 2
            self.noise_up_modules.add_module(
                "up_noize" + str(up_size),
                Up2xConv2d(middle_nc, middle_nc)
            )

        self.unet = UNetExtra(
            down_conv_count,
            in_block=nn.Sequential(
                nn.Conv2d(in_channels, gen_size, 3, stride=1, padding=1),
                nn.BatchNorm2d(gen_size),
                nn.ReLU(inplace=True)
            ),
            out_block=nn.Sequential(
                nn.Conv2d(2 * gen_size, gen_size, 3, padding=1),
                nn.BatchNorm2d(gen_size),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(gen_size, out_channels, 3, 1, 1, bias=False),
                nn.Tanh()
            ),
            middle_block=self.down_last_to_noise,
            middle_block_extra=self.noise_up_modules,
            down_block=down_block_factory,
            up_block=up_block_factory
        )

    def _forward_impl(self, noise: Tensor, condition: Tensor, *additional_input: Tensor) -> Tensor:

        return self.unet.forward(condition, noise)
