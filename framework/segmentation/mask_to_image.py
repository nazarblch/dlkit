from typing import Tuple

import torch

from torch import nn, Tensor

from framework.Loss import Loss
from framework.gan.cycle.model import CycleGAN
from framework.gan.dcgan.encoder import DCEncoder
from framework.gan.gan_model import ConditionalGANModel
from framework.gan.image2image.discriminator import Discriminator
from framework.gan.image2image.unet_generator import UNetGenerator
from framework.gan.loss.gan_loss import GANLossObject
from framework.gan.loss.hinge import HingeLoss
from framework.gan.noise.normal import NormalNoise
from framework.gan.loss.wasserstein import WassersteinLoss
from framework.segmentation.Mask import Mask
from framework.optim.min_max import MinMaxOptimizer, MinMaxLoss
from framework.parallel import ParallelConfig
from framework.segmentation.split_and_fill import weights_init
from models.networks import ResnetGenerator
from models.resnet.network import ResDiscriminator


class MaskToImage:
    def __init__(self,
                 image_size: int,
                 mask_channels_count: int,
                 image_channels_count: int = 3,
                 noise=NormalNoise(50, ParallelConfig.MAIN_DEVICE),
                 generator_size: int = 32,
                 discriminator_size: int = 32):

        self.noise = noise
        # netG = UNetGenerator(noise, image_size, mask_channels_count, image_channels_count, generator_size) \
        netG = ResnetGenerator(mask_channels_count, image_channels_count, generator_size, n_blocks=9)\
            .to(ParallelConfig.MAIN_DEVICE)
        # netD = Discriminator(discriminator_size, image_channels_count + mask_channels_count, image_size) \
        netD = ResDiscriminator(image_channels_count + mask_channels_count, discriminator_size, img_f=256, layers=4) \
            .to(ParallelConfig.MAIN_DEVICE)

        if torch.cuda.device_count() > 1:
            netD = nn.DataParallel(netD, ParallelConfig.GPU_IDS)
            netG = nn.DataParallel(netG, ParallelConfig.GPU_IDS)

        self.gan_model = ConditionalGANModel(
            netG,
            HingeLoss(netD) + GANLossObject(
                lambda x, y: Loss.ZERO(),
                lambda dgz, real, fake: Loss(nn.L1Loss()(fake[0], real[0])) * 100,
                netD
            )
        )

    def parameters(self):
        return self.gan_model.generator.parameters()

    def zero_grad(self):
        self.gan_model.generator.zero_grad()

    def train(self, images: Tensor, masks: Mask, z: Tensor):
        self.gan_model.train(images, masks.tensor, z)

    def generator_loss(self, images: Tensor, masks: Mask) -> Loss:
        z = self.noise.sample(images.shape[0])
        fake = self.gan_model.generator.forward(masks.tensor, z)
        return self.gan_model.generator_loss(images, fake, masks.tensor)

    def forward(self, masks: Mask, z: Tensor) -> Tensor:
        fake = self.gan_model.generator.forward(masks.tensor, z)
        return fake

    def __call__(self,  masks: Mask, z: Tensor) -> Tensor:
        return self.forward(masks, z)

