# Root directory for dataset
from typing import List, Tuple

import torch
from torch import nn, Tensor
from torch.distributions import Bernoulli, Categorical
from torchvision.datasets import Cityscapes
from torchvision.transforms import transforms

from data_loader.DataPath import DataPath
from data_loader.data2d.segmentation_transform import Transformer
from framework.Loss import Loss
from framework.gan.cycle.model import CycleGAN
from framework.gan.vgg.gan_loss import VggGeneratorLoss
from framework.segmentation.Mask import MaskFactory, Mask
from framework.parallel import ParallelConfig
from framework.segmentation.base import PenalizedSegmentation
from framework.segmentation.loss.entropy import SegmentationEntropy
from framework.segmentation.loss.modularity import MultiLayerModularity
from framework.segmentation.loss.neighbour_diff import NeighbourDiffLoss
from framework.segmentation.loss.sp_loss import SuperPixelsLoss
from framework.segmentation.mask_to_image import MaskToImage
from framework.segmentation.split_and_fill import SplitAndFill
from framework.segmentation.unet import UNetSegmentation
from viz.visualization import show_images, show_segmentation

# Number of workers for dataloader
workers = 10
# Batch size during training
batch_size = 10
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


dataset = Cityscapes(DataPath.HOME_STREET,
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


labels_list: List[int] = list(range(20))

segm_net = PenalizedSegmentation(
    UNetSegmentation(labels_list.__len__())
)
segm_net.add_penalty(SuperPixelsLoss())


# gan = MaskToImage(image_size, labels_list)
#
# split_gan = SplitAndFill(image_size)
#
# cycle = CycleGAN[Tensor, Tensor](
#     segm_net,
#     gan.gan_model.generator,
#     loss_1=VggGeneratorLoss(15, 1).forward,
#     loss_2=lambda s1, s2:  (
#         Loss(nn.BCELoss()(s1, s2.detach())) +
#         NeighbourDiffLoss(3)(Mask(s1)) * 2 +
#         SegmentationEntropy()(s1) * 2
#     ),
#     lr=0.0001/2
# )


def cat_sample(segm: Tensor) -> Tuple[Tensor, Mask]:

    dist = Categorical(segm.transpose(1, 2).transpose(2, 3))
    labels_sample = dist.sample()
    mask_sample: Mask = MaskFactory.from_class_map(labels_sample, list(range(segm.shape[1])))
    L: Tensor = dist.log_prob(labels_sample).mean()

    return L, mask_sample


def be_sample(segm: Tensor) -> Tuple[Tensor, Mask]:

    dist = Bernoulli(segm)
    mask_sample = dist.sample()
    L: Tensor = (segm * mask_sample + (1 - segm) * (1 - mask_sample)).log().sum() / segm.numel()

    return L,  Mask(mask_sample)


# def train_gan(imgs: Tensor):
#
#     segm: Tensor = segm_net(imgs)
#     L, mask = cat_sample(segm)
#
#     gan.train(imgs, mask)


# def train_split_gan(imgs: Tensor):
#
#     segm: Tensor = segm_net(imgs)
#     L, mask = cat_sample(segm)
#
#     segment = Transformer.get_random_segment(mask)
#     split_gan.train(imgs, segment)


# def train_segm(imgs: Tensor):
#
#     segm: Tensor = segm_net(imgs)
#
#     mod = MultiLayerModularity(5)(imgs, segm)
#     nb_diff = NeighbourDiffLoss(3)(Mask(segm)) * 2
#     entropy = SegmentationEntropy()(segm)
#
#     loss = (
#         gan.generator_loss(imgs, Mask(segm))
#         + nb_diff
#         - mod * 5
#         + entropy
#     )
#
#     print("===============================")
#     print("segm loss:" + str(loss.item()))
#     print("modularity:" + str(mod.item()))
#     print("nb diff:" + str(nb_diff.item()))
#     print("entropy:" + str(entropy.item()))
#
#     loss.minimize_step(segm_net.opt)
#
#     segm: Tensor = segm_net(imgs)
#     L, mask = cat_sample(segm)
#     cycle.train(imgs, mask.tensor)
#
#     segm: Tensor = segm_net(imgs)
#     segment = Transformer.get_random_segment(Mask(segm))
#     split_loss = (
#         split_gan.generator_loss(imgs, segment) +
#         NeighbourDiffLoss(3)(Mask(segm)) * 2 +
#         SegmentationEntropy()(segm)
#     )
#
#     split_loss.minimize_step(segm_net.opt)


def train_sup_segm(imgs: Tensor, mask: Mask):

    segm: Tensor = segm_net(imgs)

    loss = Loss(nn.BCELoss()(segm, mask.tensor))

    loss.minimize_step(segm_net.opt)


# def show_fake_and_segm(imgs: Tensor):
#
#     with torch.no_grad():
#         segm: Tensor = segm_net(imgs[:16])
#         fake = gan.gan_model.generator(segm).detach()
#         show_images(fake.cpu(), 4, 4)
#         show_segmentation(segm.cpu())
#
#
# def show_split_segm(imgs: Tensor):
#
#     with torch.no_grad():
#         segm: Tensor = segm_net(imgs[:16])
#         segment = Transformer.get_random_segment(Mask(segm))
#         front, bk = split_gan.test(imgs[:16], segment)
#         show_images(front.cpu(), 4, 4)
#         show_images(bk.cpu(), 4, 4)
#         show_segmentation(segm.cpu())


print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, (imgs, labels) in enumerate(dataloader, 0):

        imgs = imgs.to(ParallelConfig.MAIN_DEVICE)

        segm_net.train(imgs)

        # train_gan(imgs)
        # train_split_gan(imgs)
        # train_segm(imgs)

        if i % 10 == 0:
            print(i)
            show_segmentation(segm_net.forward(imgs).cpu())
            # show_split_segm(imgs)
            # show_fake_and_segm(imgs)







