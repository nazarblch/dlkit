from torch import nn, Tensor

from framework.gan.discriminator import Discriminator as D


class DCDiscriminator(D):
    def __init__(self):
        super(DCDiscriminator, self).__init__()
        nc = 3
        ndf = 64

        self.main = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.InstanceNorm2d(ndf, affine=True),
            # input is (ndf) x 64 x 64
            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.InstanceNorm2d(ndf, affine=True),
            # state size. (ndf) x 32 x 32
            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.InstanceNorm2d(ndf * 2, affine=True),
            # state size. (ndf*2) x 16 x 16
            nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.InstanceNorm2d(ndf * 4, affine=True),
            # state size. (ndf*4) x 8 x 8
            nn.utils.spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.InstanceNorm2d(ndf * 8, affine=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.main(x)
