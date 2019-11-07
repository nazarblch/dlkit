from typing import List

from torch import Tensor, nn
from torch.nn import init

from framework.Loss import Loss
from framework.gan.loss.gan_loss import GANLoss
from framework.gan.conditional import ConditionalGenerator, ConditionalDiscriminator
from framework.gan.discriminator import Discriminator
from framework.gan.generator import Generator
from framework.optim.min_max import MinMaxParameters, MinMaxLoss, MinMaxOptimizer


# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         nn.init.normal_(m.weight.data, 1.0, 0.02)
#         nn.init.constant_(m.bias.data, 0)

def weights_init(net, init_type='normal', gain=0.02):
    """Get different initial method for the network weights"""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv')!=-1 or classname.find('Linear')!=-1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class GANModel:

    def __init__(self, generator: Generator, loss: GANLoss, lr=0.0002):
        self.generator = generator
        self.generator.apply(weights_init)
        self.loss = loss
        self.loss.discriminator.apply(weights_init)
        params = MinMaxParameters(self.generator.parameters(), self.loss.parameters())
        self.optimizer = MinMaxOptimizer(params, lr, lr * 2)

    def discriminator_loss(self, real: Tensor, fake: Tensor) -> Loss:
        return self.loss.discriminator_loss_with_penalty([real], [fake])

    def generator_loss(self, real: Tensor, fake: Tensor) -> Loss:
        return self.loss.generator_loss([real], [fake])

    def loss_pair(self, real: Tensor, *noise: Tensor) -> MinMaxLoss:

        fake = self.generator.forward(*noise)

        return MinMaxLoss(
            self.generator_loss(real, fake),
            self.discriminator_loss(real, fake)
        )

    def parameters(self) -> MinMaxParameters:
        return MinMaxParameters(self.generator.parameters(), self.loss.parameters())

    def forward(self, real: Tensor, *noise: Tensor):
        return self.loss_pair(real, *noise)

    def train(self, real: Tensor, *noise: Tensor):
        loss = self.loss_pair(real, *noise)
        self.optimizer.train_step(loss)
        return loss.min_loss.item(), loss.max_loss.item()


class ConditionalGANModel:

    def __init__(self, generator: ConditionalGenerator, loss: GANLoss, lr=0.0002):
        self.generator = generator
        self.loss = loss
        params = MinMaxParameters(self.generator.parameters(), self.loss.parameters())
        self.optimizer = MinMaxOptimizer(params, lr, lr * 2)

    def discriminator_loss(self, real: Tensor, fake: Tensor, condition: Tensor) -> Loss:
        return self.loss.discriminator_loss_with_penalty([real, condition], [fake, condition])

    def generator_loss(self, real: Tensor, fake: Tensor, condition: Tensor) -> Loss:
        return self.loss.generator_loss([real, condition], [fake, condition])

    def loss_pair(self, real: Tensor, condition: Tensor, *noise: Tensor) -> MinMaxLoss:
        fake = self.generator.forward(condition, *noise)
        return MinMaxLoss(
            self.generator_loss(real, fake, condition),
            self.discriminator_loss(real, fake, condition)
        )

    def parameters(self) -> MinMaxParameters:
        return MinMaxParameters(self.generator.parameters(), self.loss.parameters())

    def forward(self, real: Tensor, condition: Tensor, *noise: Tensor):
        return self.loss_pair(real, condition, *noise)

    def train(self, real: Tensor, condition: Tensor, *noise: Tensor):
        loss = self.loss_pair(real, condition, *noise)
        self.optimizer.train_step(loss)
        return loss.min_loss.item(), loss.max_loss.item()
