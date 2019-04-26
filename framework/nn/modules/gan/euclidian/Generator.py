from torch import nn, Tensor

from framework.nn.modules.gan.Generator import Generator as G
from framework.nn.modules.gan.noise.Noise import Noise


class Generator(G):

    def __init__(self, noise: Noise):
        super(Generator, self).__init__(noise)
        n_out = 2
        ngf = 32
        self.main = nn.Sequential(
            nn.Linear(noise.size, ngf),
            nn.ReLU(True),
            nn.Linear(ngf,  2 * ngf),
            nn.Tanh(),
            nn.Linear(2 * ngf, n_out)
        )

    def _forward_impl(self, noise: Tensor) -> Tensor:
        return self.main(noise)
