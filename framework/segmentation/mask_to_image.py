import torch
from typing import List

from torch import nn, Tensor

from framework.Loss import Loss
from framework.nn.modules.gan.GANModel import ConditionalGANModel
from framework.nn.modules.gan.image2image.discriminator import Discriminator
from framework.nn.modules.gan.image2image.unet_generator import UNetGenerator
from framework.nn.modules.gan.noise.normal import NormalNoise
from framework.nn.modules.gan.penalties.AdaptiveLipschitzPenalty import AdaptiveLipschitzPenalty
from framework.nn.modules.gan.penalties.l2_penalty import L2Penalty
from framework.nn.modules.gan.vgg.gan_loss import VggGeneratorLoss
from framework.nn.modules.gan.wgan.WassersteinLoss import WassersteinLoss
from framework.nn.ops.segmentation.Mask import Mask
from framework.optim.min_max import MinMaxOptimizer, MinMaxLoss
from framework.parallel import ParallelConfig
from framework.segmentation.split_and_fill import weights_init


class MaskToImage:
    def __init__(self,
                 image_size: int,
                 labels_list: List[int],
                 image_channels_count: int = 3,
                 noise=NormalNoise(100, ParallelConfig.MAIN_DEVICE),
                 generator_size: int = 64,
                 discriminator_size: int = 64):

        mask_channels = len(labels_list)

        netG = UNetGenerator(noise, image_size, mask_channels, image_channels_count, generator_size) \
            .to(ParallelConfig.MAIN_DEVICE)
        netD = Discriminator(discriminator_size, image_channels_count + mask_channels, image_size) \
            .to(ParallelConfig.MAIN_DEVICE)

        netG.apply(weights_init)
        netD.apply(weights_init)

        if torch.cuda.device_count() > 1:
            netD = nn.DataParallel(netD, ParallelConfig.GPU_IDS)
            netG = nn.DataParallel(netG, ParallelConfig.GPU_IDS)

        self.gan_model = ConditionalGANModel(
            netG,
            netD,
            WassersteinLoss(2)
                .add_penalty(AdaptiveLipschitzPenalty(0.1, 0.01))
                .add_penalty(L2Penalty(1))  # + VggGeneratorLoss(0.5)
        )

        # vgg_loss_fn = VggGeneratorLoss(ParallelConfig.MAIN_DEVICE)

        lrG = 0.0001
        lrD = 0.0001
        self.optimizer = MinMaxOptimizer(self.gan_model.parameters(), lrG, lrD)

    def train(self, images: Tensor, masks: Mask):

        loss: MinMaxLoss = self.gan_model.loss_pair(images, masks.data)
        self.optimizer.train_step(loss)

    def generator_loss(self, images: Tensor, masks: Mask) -> Loss:

        fake = self.gan_model.generator.forward(masks.data)
        return self.gan_model.generator_loss(images, fake, masks.data)

