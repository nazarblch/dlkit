from typing import Tuple

import torch

from torch import nn, Tensor

from framework.Loss import Loss
from framework.nn.modules.gan.GANModel import ConditionalGANModel
from framework.nn.modules.gan.image2image.discriminator import Discriminator
from framework.nn.modules.gan.image2image.unet_generator import UNetGenerator
from framework.nn.modules.gan.noise.Noise import Noise
from framework.nn.modules.gan.noise.normal import NormalNoise
from framework.nn.modules.gan.penalties.AdaptiveLipschitzPenalty import AdaptiveLipschitzPenalty
from framework.nn.modules.gan.penalties.l2_penalty import L2Penalty
from framework.nn.modules.gan.vgg.gan_loss import VggGeneratorLoss
from framework.nn.modules.gan.wgan.WassersteinLoss import WassersteinLoss
from framework.nn.ops.segmentation.Mask import Mask
from framework.optim.min_max import MinMaxOptimizer, MinMaxLoss
from framework.parallel import ParallelConfig


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        except:
            {}
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class FillGenerator(UNetGenerator):

    def _forward_impl(self, noise: Tensor, condition: Tensor, *additional_input: Tensor) -> Tensor:

        fake = self.unet.forward(condition, noise)
        segment = additional_input[0]
        return fake * (1 - segment) + segment * condition[:, 0:3, :, :]


class SplitAndFill:

    def __init__(self,
                 image_size: int,
                 generator_size: int = 64,
                 discriminator_size: int = 64,
                 channels_count: int = 3):

        self.noise = NormalNoise(100, ParallelConfig.MAIN_DEVICE)

        self.bkG = FillGenerator(self.noise, image_size, channels_count + 1, channels_count, generator_size)\
            .to(ParallelConfig.MAIN_DEVICE)
        self.frontG = FillGenerator(self.noise, image_size, channels_count + 1, channels_count, generator_size)\
            .to(ParallelConfig.MAIN_DEVICE)
        self.D = Discriminator(discriminator_size, 2 * channels_count + 1, image_size)\
            .to(ParallelConfig.MAIN_DEVICE)

        self.bkG.apply(weights_init)
        self.frontG.apply(weights_init)
        self.D.apply(weights_init)

        if ParallelConfig.GPU_IDS.__len__() > 1:
            self.bkG = nn.DataParallel(self.bkG, ParallelConfig.GPU_IDS)
            self.frontG = nn.DataParallel(self.frontG, ParallelConfig.GPU_IDS)
            self.D = nn.DataParallel(self.D, ParallelConfig.GPU_IDS)

        was_loss = WassersteinLoss(2)\
            .add_penalty(AdaptiveLipschitzPenalty(0.1, 0.01))\
            .add_penalty(L2Penalty(0.1))  # + VggGeneratorLoss(0.5)

        self.front_gan_model = ConditionalGANModel(self.frontG, self.D, was_loss)
        self.bk_gan_model = ConditionalGANModel(self.bkG, self.D, was_loss)

        lr = 0.0001
        self.optimizer_front = MinMaxOptimizer(self.front_gan_model.parameters(), lr, lr)
        self.optimizer_bk = MinMaxOptimizer(self.bk_gan_model.parameters(), lr, lr)

    def train(self, images: Tensor, segments: Mask):

        front: Tensor = torch.cat((images * segments.data, 1 - segments.data), dim=1)
        loss_front: MinMaxLoss = self.front_gan_model.loss_pair(images, front, segments.data)
        self.optimizer_front.train_step(loss_front)

        bk: Tensor = torch.cat((images * (1 - segments.data), segments.data), dim=1)
        loss_bk: MinMaxLoss = self.bk_gan_model.loss_pair(images, bk, 1 - segments.data)
        self.optimizer_bk.train_step(loss_bk)

    def test(self, images: Tensor, segments: Mask) -> Tuple[Tensor, Tensor]:

        front: Tensor = torch.cat((images * segments.data, 1 - segments.data), dim=1)
        bk: Tensor = torch.cat((images * (1 - segments.data), segments.data), dim=1)

        return self.frontG(front, segments.data), self.bkG(bk, 1 - segments.data)

    def generator_loss(self, images: Tensor, segments: Mask) -> Loss:

        front: Tensor = torch.cat((images * segments.data, 1 - segments.data), dim=1)
        front_fake = self.frontG(front, segments.data)
        loss_front = self.front_gan_model.generator_loss(images, front_fake, front)

        bk: Tensor = torch.cat((images * (1 - segments.data), segments.data), dim=1)
        bk_fake = self.bkG(bk, 1 - segments.data)
        loss_bk = self.bk_gan_model.generator_loss(images, bk_fake, bk)

        return loss_bk + loss_front

