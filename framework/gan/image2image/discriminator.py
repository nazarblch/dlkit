import torch
from torch import nn, Tensor
import numpy as np

from framework.gan.conditional import ConditionalDiscriminator
from framework.nn.modules.common.View import View


class Discriminator(ConditionalDiscriminator):
    def __init__(self, ndf, nc, img_size):
        super(Discriminator, self).__init__()

        down_times = int(np.log2(img_size))
        assert 2**down_times == img_size

        max_nc = 512

        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.1, inplace=True)
        )

        size = int(img_size / 2)
        ndf_tmp = ndf

        for di in range(down_times):
            if size == 2:
                break
            nc_out = min(2 * ndf_tmp, max_nc)
            self.main.add_module("disc_down_" + str(size), nn.Sequential(
                nn.Conv2d(ndf_tmp, nc_out, 4, 2, 1, bias=False),
                nn.BatchNorm2d(nc_out),
                nn.LeakyReLU(0.1, inplace=True)
            ))
            size = int(size / 2)
            ndf_tmp = min(ndf_tmp * 2, max_nc)

        # self.main.add_module("disc_down_final", nn.Sequential(
        #     nn.Conv2d(ndf_tmp, 1, 4, 1, 0, bias=False)
        # ))

        out = nn.Sequential(
            View(-1, 2 * 2 * ndf_tmp),
            nn.Linear(2 * 2 * ndf_tmp, ndf_tmp),
            nn.ReLU(inplace=True),
            nn.Linear(ndf_tmp, 1)
        )

        self.main.add_module("out", out)

    def forward(self, image: Tensor, *mask: Tensor) -> Tensor:
        return self.main(torch.cat((image, mask[0]), dim=1))
