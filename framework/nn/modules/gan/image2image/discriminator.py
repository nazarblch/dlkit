from torch import nn, Tensor

from framework.nn.modules.gan.ConditionalDiscriminator import ConditionalDiscriminator


class Discriminator(ConditionalDiscriminator):
    def __init__(self, ndf, nc, img_size):
        # TODO: involve img_size!
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ndf, ndf, 3, stride=1, padding=1),
            nn.BatchNorm2d(ndf, 0.8),
            nn.ReLU(inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(ndf * 2, 0.8),
            nn.ReLU(inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 4, 3, stride=1, padding=1),
            nn.BatchNorm2d(ndf * 4, 0.8),
            nn.ReLU(inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 8),
            # nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Dropout(),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
        )

    def forward(self, image: Tensor, mask: Tensor) -> Tensor:
        return self.main(input)
