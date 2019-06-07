# Root directory for dataset
from typing import List

import torch
from torch import nn, Tensor
from torch.distributions import Bernoulli
from torchvision.datasets import Cityscapes
from torchvision.transforms import transforms

from data_loader.data2d.segmentation_transform import Transformer
from framework.Loss import Loss
from framework.nn.modules.gan.GANModel import ConditionalGANModel
from framework.nn.modules.gan.image2image.gan_factory import MaskToImageFactory
from framework.optim.min_max import MinMaxOptimizer, MinMaxLoss
from framework.nn.modules.gan.penalties.AdaptiveLipschitzPenalty import AdaptiveLipschitzPenalty
from framework.nn.modules.gan.penalties.l2_penalty import L2Penalty
from framework.nn.modules.gan.vgg.gan_loss import VggGeneratorLoss
from framework.nn.modules.gan.wgan.WassersteinLoss import WassersteinLoss
from framework.nn.ops.segmentation.Mask import MaskFactory, Mask
from framework.parallel import ParallelConfig
from framework.segmentation.mask_to_image import MaskToImage
from framework.segmentation.split_and_fill import SplitAndFill
from framework.segmentation.unet import UNetSegmentation
from viz.visualization import show_images, show_segmentation
import torch.nn.functional as F

# Number of workers for dataloader
workers = 20
# Batch size during training
batch_size = 32
# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 128
# Number of channels in the training images. For color images this is 3
nc = 3
# Size of z latent vector (i.e. size of generator input)
nz = 100
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64
# Number of training epochs
num_epochs = 30


dataroot = "/gpfs/gpfs0/n.buzun/segmentation_data"
dataset = Cityscapes(dataroot,
                     transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ]),
                    target_transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor()
                    ])
                    )

# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=workers)


labels_list: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 33]

segm_net = nn.DataParallel(
    UNetSegmentation(labels_list.__len__()).to(ParallelConfig.MAIN_DEVICE),
    ParallelConfig.GPU_IDS
)

opt = torch.optim.Adam(segm_net.parameters(), lr=0.0003, betas=(0.5, 0.999))


gan = MaskToImage(image_size, labels_list)

split_gan = SplitAndFill(image_size)


neighbour_filter: Tensor = torch.zeros(4, 1, 3, 3, dtype=torch.float32).to(ParallelConfig.MAIN_DEVICE)
neighbour_filter[:, 0, 1, 1] = 1
neighbour_filter[0, 0, 0, 1] = -1
neighbour_filter[1, 0, 1, 0] = -1
neighbour_filter[2, 0, 1, 2] = -1
neighbour_filter[3, 0, 2, 1] = -1


def neighbour_diff_loss(segm: Tensor) -> Loss:

    res = 0

    for segm_i in segm.split(1, dim=1):
        res += F.conv2d(segm_i, neighbour_filter).abs().mean()

    return Loss(res / segm.shape[1])


def train_gan(imgs: Tensor):

    segm_out: Tensor = segm_net(imgs)
    gan.train(imgs, Mask(
        Bernoulli(segm_out).sample()
    ))


def train_split_gan(imgs: Tensor):

    segm: Tensor = segm_net(imgs)
    mask = Mask(Bernoulli(segm).sample())

    segment = Transformer.get_random_segment(mask)
    split_gan.train(imgs, segment)


def train_segm(imgs: Tensor):

    segm: Tensor = segm_net(imgs)
    sample = Bernoulli(segm).sample()
    L = (segm * sample + (1 - segm) * (1 - sample)).log().sum() / segm.numel()
    loss = gan.generator_loss(imgs, Mask(segm)) - Loss(L) / 10 + neighbour_diff_loss(segm) * 20

    opt.zero_grad()
    loss.minimize()
    opt.step()

    print("segm loss:" + str(loss.item()))

    segm: Tensor = segm_net(imgs)
    fake = gan.gan_model.generator.forward(segm)
    fake_segm = segm_net(fake)
    cycle_loss = Loss(
        nn.BCELoss()(fake_segm, segm.detach())
        + nn.BCELoss()(segm, fake_segm.detach())
    )

    opt.zero_grad()
    cycle_loss.minimize()
    opt.step()

    segm: Tensor = segm_net(imgs)
    segment = Transformer.get_random_segment(segm)
    split_loss = split_gan.generator_loss(imgs, segment)

    opt.zero_grad()
    split_loss.minimize()
    opt.step()


def show_fake_and_segm(imgs: Tensor):

    with torch.no_grad():
        segm: Tensor = segm_net(imgs)
        fake = gan.gan_model.generator(segm).detach()
        show_images(fake.cpu(), 4, 4)
        show_segmentation(segm.cpu())


print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, (imgs, labels) in enumerate(dataloader, 0):

        imgs = imgs.to(ParallelConfig.MAIN_DEVICE)
        # mask = MaskFactory.from_class_map(labels.to(ParallelConfig.MAIN_DEVICE), labels_list)

        train_gan(imgs)
        train_split_gan(imgs)
        train_segm(imgs)

        if i % 20 == 0:
            show_fake_and_segm(imgs)







