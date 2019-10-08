from torch import nn, Tensor

from framework.gan.generator import Generator as G
from framework.gan.noise import Noise


class EGenerator(G):

    def _device(self):
        return next(self.main.parameters()).device

    def __init__(self, noise: Noise):
        super(EGenerator, self).__init__(noise)
        n_out = 2
        ngf = 32
        self.main = nn.Sequential(
            nn.Linear(noise.size(), ngf),
            nn.ReLU(True),
            nn.Linear(ngf,  2 * ngf),
            nn.Tanh(),
            nn.Linear(2 * ngf, n_out)
        )

    def _forward_impl(self, noise: Tensor) -> Tensor:
        return self.main(noise)
