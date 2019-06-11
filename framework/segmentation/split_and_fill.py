from typing import Tuple

from torch import nn, Tensor

from framework.Loss import Loss
from framework.gan.image2image.discriminator import Discriminator
from framework.nn.modules.common.vgg import Vgg16
from framework.gan.GANModel import ConditionalGANModel
from framework.gan.image2image.unet_generator import UNetGenerator
from framework.gan.noise.normal import NormalNoise
from framework.gan.penalties.AdaptiveLipschitzPenalty import AdaptiveLipschitzPenalty
from framework.gan.penalties.l2_penalty import L2Penalty
from framework.gan.wgan.WassersteinLoss import WassersteinLoss
from framework.segmentation.Mask import Mask
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


class VggDotLoss(nn.Module):

    def __init__(self, depth: int = 15, weight: float = 1):
        super(VggDotLoss, self).__init__()
        self.vgg = Vgg16(depth).to(ParallelConfig.MAIN_DEVICE)
        if ParallelConfig.GPU_IDS.__len__() > 1:
            self.vgg = nn.DataParallel(self.vgg, ParallelConfig.GPU_IDS)

        self.weight = weight

    def forward(self, x: Tensor, y: Tensor) -> Loss:

        x_vgg, y_vgg = self.vgg(x).view(x.size(0), -1), self.vgg(y).view(x.size(0), -1)
        x_vgg = x_vgg - x_vgg.mean(dim=0, keepdim=True).detach()
        y_vgg = y_vgg - y_vgg.mean(dim=0, keepdim=True).detach()

        return Loss(
            (x_vgg * y_vgg).mean()
        )


class FillGenerator(UNetGenerator):

    def _forward_impl(self, noise: Tensor, condition: Tensor, *additional_input: Tensor) -> Tensor:

        fake = self.unet.forward(condition, noise)
        segment = additional_input[0]
        return fake * (1 - segment) + segment * condition


class SplitAndFill:

    def __init__(self,
                 image_size: int,
                 generator_size: int = 32,
                 discriminator_size: int = 32,
                 channels_count: int = 3):

        self.noise = NormalNoise(100, ParallelConfig.MAIN_DEVICE)

        self.bkG = FillGenerator(self.noise, image_size, channels_count, channels_count, generator_size)\
            .to(ParallelConfig.MAIN_DEVICE)
        self.frontG = FillGenerator(self.noise, image_size, channels_count, channels_count, generator_size)\
            .to(ParallelConfig.MAIN_DEVICE)
        self.D = Discriminator(discriminator_size, 2 * channels_count, image_size)\
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
            .add_penalty(L2Penalty(0.1))

        self.front_gan_model = ConditionalGANModel(self.frontG, self.D, was_loss)
        self.bk_gan_model = ConditionalGANModel(self.bkG, self.D, was_loss)

        lr = 0.0002
        self.optimizer_front = MinMaxOptimizer(self.front_gan_model.parameters(), lr, lr)
        self.optimizer_bk = MinMaxOptimizer(self.bk_gan_model.parameters(), lr, lr)

        # self.vgg_dot = VggDotLoss(20, 0.5)

    def train(self, images: Tensor, segments: Mask):

        front: Tensor = images * segments.tensor
        loss_front: MinMaxLoss = self.front_gan_model.loss_pair(images, front, segments.tensor)
        self.optimizer_front.train_step(loss_front)

        bk: Tensor = images * (1 - segments.tensor)
        loss_bk: MinMaxLoss = self.bk_gan_model.loss_pair(images, bk, 1 - segments.tensor)
        self.optimizer_bk.train_step(loss_bk)

    def test(self, images: Tensor, segments: Mask) -> Tuple[Tensor, Tensor]:

        front: Tensor = images * segments.tensor
        bk: Tensor = images * (1 - segments.tensor)

        return self.frontG(front, segments.tensor), self.bkG(bk, 1 - segments.tensor)

    def generator_loss(self, images: Tensor, segments: Mask) -> Loss:

        front: Tensor = images * segments.tensor
        front_fake = self.frontG(front, segments.tensor)
        loss_front = self.front_gan_model.generator_loss(images, front_fake, front)

        bk: Tensor = images * (1 - segments.tensor)
        bk_fake = self.bkG(bk, 1 - segments.tensor)
        loss_bk = self.bk_gan_model.generator_loss(images, bk_fake, bk)

        # dot_loss = self.vgg_dot(images * segments.tensor, images * (1 - segments.tensor))

        return loss_bk + loss_front  # + dot_loss

