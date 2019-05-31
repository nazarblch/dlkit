import torch
from torch import Tensor, nn

from framework.Loss import Loss
from framework.nn.modules.common.vgg import Vgg19
from framework.nn.modules.gan.GANLoss import GANLoss


class VggGeneratorLoss(GANLoss, nn.Module):

    def __init__(self, dev: torch.device):
        super(VggGeneratorLoss, self).__init__()
        self.vgg = Vgg19().to(dev)
        self.criterion = nn.L1Loss()
        self.weights = [1.0/4, 1.0/2, 1.0]
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)

    def forward(self, x: Tensor, y: Tensor) -> Loss:
        while x.size()[3] > 1024:
            x, y = self.downsample(x), self.downsample(y)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss: float = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i])
        return Loss(loss)

    def generator_loss(self, dgz: Tensor, real: Tensor, fake: Tensor) -> Loss:
        return self.forward(fake, real)

    def discriminator_loss(self, d_real: Tensor, d_fake: Tensor) -> Loss:
        return Loss.ZERO()
