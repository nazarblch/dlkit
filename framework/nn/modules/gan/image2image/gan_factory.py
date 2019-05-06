from typing import Tuple, List

from torch import nn
from torch import device as Device

from framework.nn.modules.gan.ConditionalDiscriminator import ConditionalDiscriminator
from framework.nn.modules.gan.ConditionalGenerator import ConditionalGenerator
from framework.nn.modules.gan.image2image.discriminator import Discriminator
from framework.nn.modules.gan.image2image.unet_generator import UNetGenerator
from framework.nn.modules.gan.noise.Noise import Noise
from framework.nn.modules.gan.noise.normal import NormalNoise


def GANFactory(image_size: int,
               noise_size: int,
               generator_size: int,
               discriminator_size: int,
               out_chanels_count: int,
               device: Device,
               labels_list: List[int]) -> Tuple[ConditionalGenerator, ConditionalDiscriminator]:

    # custom weights initialization called on netG and netD
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

    mask_channels = len(labels_list)
    noise = NormalNoise(noise_size, device)
    netG = UNetGenerator(noise, image_size, mask_channels, out_chanels_count, generator_size).to(device)
    netD = Discriminator(discriminator_size, out_chanels_count + mask_channels, image_size).to(device)

    # # Handle multi-gpu if desired
    # ngpu = torch.cuda.device_count()
    # gpu_ids = [1, 2, 3]
    # if (device.type == 'cuda') and (ngpu > 1):
    #     netG = nn.DataParallel(netG, gpu_ids)
    #     netD = nn.DataParallel(netD, gpu_ids)

    netG.apply(weights_init)
    netD.apply(weights_init)

    return netG, netD