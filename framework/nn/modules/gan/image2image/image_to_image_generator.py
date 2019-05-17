from functools import reduce
from typing import List

import torch
from torch import nn, Tensor

from framework.nn.modules.common.Down2xConv2d import Down2xConv2d
from framework.nn.modules.common.Up2xConv2d import Up4xConv2d, Up2xConv2d
from framework.nn.modules.common.View import View
from framework.nn.modules.gan.ConditionalGenerator import ConditionalGenerator as CG
from framework.nn.modules.gan.noise.Noise import Noise


class UNetGenerator(CG):

    def __init__(self, noise: Noise, image_size, in_channels, out_channels, gen_size=64, down_conv_count=4):

        super(UNetGenerator, self).__init__(noise)

        middle_data_size = int(image_size * 2**(-down_conv_count))

        assert (middle_data_size >= 4)
        assert (middle_data_size % 4 == 0)

        # TODO: create DownBlocksList
        self.down_modules = nn.ModuleList([
            Down2xConv2d(in_channels, gen_size)
        ])

        module_size = gen_size
        for i in range(0, down_conv_count - 1):
            self.down_modules.append(Down2xConv2d(module_size, 2 * module_size))
            module_size *= 2

        self.down_last_to_noise = nn.Sequential(
            View(-1, module_size * middle_data_size**2),
            nn.Linear(module_size * middle_data_size**2, noise.size()),
            # nn.BatchNorm1d(noise.size(), 0.9),
            nn.Tanh()
        )

        self.noise_up_modules = nn.Sequential(
            nn.Linear(2 * noise.size(), 2 * noise.size()),
            # nn.BatchNorm1d(2 * noise.size(), 0.9),
            nn.Tanh(),
            nn.Linear(2 * noise.size(), 2 * noise.size()),
            # nn.BatchNorm1d(2 * noise.size(), 0.9),
            nn.Tanh(),
            View(-1, 2 * noise.size(), 1, 1),
            Up4xConv2d(2 * noise.size(), module_size)
        )

        up_size = 4

        while up_size < middle_data_size:
            up_size *= 2
            self.noise_up_modules.add_module(
                "up_noize" + str(up_size),
                Up2xConv2d(module_size, module_size)
            )

        self.common_up_modules = nn.ModuleList([Up2xConv2d(2 * module_size, int(module_size))])
        up_size *= 2
        module_size = int(module_size / 2)

        while up_size < image_size:
            up_size *= 2
            self.common_up_modules.append(
                Up2xConv2d(3 * module_size, int(module_size))
            )
            module_size = int(module_size / 2)

        assert len(self.common_up_modules) == len(self.down_modules)

        self.gen_out = nn.Sequential(
            nn.Conv2d(2 * module_size, module_size, 3, padding=1),
            nn.BatchNorm2d(module_size),
            nn.Dropout(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(module_size, out_channels, 3, 1, 1, bias=False)
        )

    def _forward_impl(self, noise: Tensor, mask: Tensor) -> Tensor:

        down_list: List[Tensor] = [mask]

        for layer in self.down_modules:
            down_list.append(layer(down_list[-1]))
        down_list.__delitem__(0)

        zm: Tensor = self.down_last_to_noise(down_list[-1])

        z = nn.Dropout().forward(
            self.noise_up_modules(
                torch.cat((noise, zm), dim=1)
            )
        )

        i = len(down_list) - 1
        for layer in self.common_up_modules:
            z = layer(torch.cat((z, down_list[i]), dim=1))
            i -= 1

        z = self.gen_out(z)

        return z
