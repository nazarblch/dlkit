from torch import nn, Tensor

from framework.gan.discriminator import Discriminator as D


class EDiscriminator(D):
    def __init__(self, dim=2, ndf=64):
        super(EDiscriminator, self).__init__()

        self.main = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(dim, ndf)),
            nn.ReLU(True),
            nn.utils.spectral_norm(nn.Linear(ndf, 2 * ndf)),
            nn.ReLU(True),
            nn.utils.spectral_norm(nn.Linear(2 * ndf, 2 * ndf)),
            nn.ReLU(True),
            nn.utils.spectral_norm(nn.Linear(2 * ndf, 1))
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.main(x)
