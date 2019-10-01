import math

from torch import Tensor, nn

from framework.gan.generator import Generator as G
from framework.gan.noise import Noise


class DCGenerator(G):

    def __init__(self, noise: Noise, image_size: int, ngf=64):
        super(DCGenerator, self).__init__(noise)
        n_up = int(math.log2(image_size / 4))
        assert 4 * (2 ** n_up) == image_size
        nc = 3

        layers = [
            nn.utils.spectral_norm(nn.ConvTranspose2d(noise.size(), ngf * 8, 4, 1, 0, bias=False)),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
        ]

        nc_l_next = -1
        for l in range(n_up):

            nc_l = max(ngf, (ngf * 8) // 2**l)
            nc_l_next = max(ngf, nc_l // 2)

            layers += [
                nn.utils.spectral_norm(nn.ConvTranspose2d(nc_l, nc_l_next, 4, stride=2, padding=1, bias=False)),
                nn.BatchNorm2d(nc_l_next),
                nn.ReLU(True),
            ]

        layers += [
            nn.Conv2d(nc_l_next, nc, 3, 1, 1, bias=True)
        ]

        self.main = nn.Sequential(*layers)

    def _forward_impl(self, noise: Tensor) -> Tensor:
        return self.main(noise.view(*noise.size(), 1, 1))
