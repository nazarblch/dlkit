import torch
from torch import Tensor, nn

from framework.nn.modules.common.View import View
from framework.nn.modules.gan.Generator import Generator as G


class Generator(G):

    def __init__(self, batch_size: int, noise_size: int, device: torch.device):
        super(Generator, self).__init__(batch_size, noise_size, device)
        nc = 3
        ngf = 64
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            View(batch_size, noise_size, 1, 1),
            nn.ConvTranspose2d(noise_size, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
        super(Generator, self).to(device)

    def forward(self, noise: Tensor) -> Tensor:
        return self.main(noise)
