import torch
from torch import nn, Tensor
import numpy as np
from framework.nn.modules.gan.ConditionalDiscriminator import ConditionalDiscriminator


class Discriminator(ConditionalDiscriminator):
    def __init__(self, ndf, nc, img_size):
        super(Discriminator, self).__init__()

        down_times = int(np.log2(img_size))
        assert 2**down_times == img_size

        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ndf, ndf, 3, stride=1, padding=1),
            nn.BatchNorm2d(ndf, 0.8),
            nn.ReLU(inplace=True)
        )

        size = int(img_size / 2)
        ndf_tmp = ndf

        for di in range(down_times):
            if size == 4:
                break
            self.main.add_module("disc_down_" + str(size), nn.Sequential(
                nn.Conv2d(ndf_tmp, 2 * ndf_tmp, 4, 2, 1, bias=False),
                nn.BatchNorm2d(2 * ndf_tmp),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(2 * ndf_tmp, 2 * ndf_tmp, 3, stride=1, padding=1),
                nn.BatchNorm2d(2 * ndf_tmp),
                nn.ReLU(inplace=True)
            ))
            size = int(size / 2)
            ndf_tmp = ndf_tmp * 2

        self.main.add_module("disc_down_final", nn.Sequential(
            nn.Conv2d(ndf_tmp, 1, 4, 1, 0, bias=False)
        ))

    def forward(self, image: Tensor, *mask: Tensor) -> Tensor:
        return self.main(torch.cat((image, mask[0]), dim=1))
