from torch import nn, Tensor

from framework.gan import discriminator as D


class Discriminator(D):
    def __init__(self):
        super(Discriminator, self).__init__()
        dim = 2
        ndf = 64

        self.main = nn.Sequential(
            nn.Linear(dim, ndf),
            nn.ReLU(True),
            nn.Linear(ndf, 2 * ndf),
            nn.ReLU(True),
            nn.Linear(2 * ndf, 2 * ndf),
            nn.ReLU(True),
            nn.Linear(2 * ndf, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.main(x)
