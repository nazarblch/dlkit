# Root directory for dataset
from typing import List, Tuple

import torch
from torch import nn, Tensor
from torch.distributions import Bernoulli, Categorical
from torchvision.datasets import Cityscapes
from torchvision.transforms import transforms

from data_loader.data2d.segmentation_transform import Transformer
from framework.Loss import Loss
from framework.nn.ops.segmentation.Mask import MaskFactory, Mask
from framework.parallel import ParallelConfig
from framework.segmentation.loss.neighbour_diff import NeighbourDiffLoss
from framework.segmentation.mask_to_image import MaskToImage
from framework.segmentation.segment_to_image import MaskToImageComposite
from framework.segmentation.split_and_fill import SplitAndFill
from framework.segmentation.unet import UNetSegmentation
from viz.visualization import show_images, show_segmentation

# Number of workers for dataloader
workers = 20
# Batch size during training
batch_size = 64
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


labels_list: List[int] = [1, 2, 3, 4, 5]

segm_net = nn.DataParallel(
    UNetSegmentation(labels_list.__len__()).to(ParallelConfig.MAIN_DEVICE),
    ParallelConfig.GPU_IDS
)

opt = torch.optim.Adam(segm_net.parameters(), lr=0.0002, betas=(0.5, 0.999))


gan = MaskToImage(image_size, labels_list)

# split_gan = SplitAndFill(image_size)


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


def train_gan(imgs: Tensor):

    segm: Tensor = segm_net(imgs)
    L, mask = be_sample(segm)

    gan.train(imgs, mask)


def train_split_gan(imgs: Tensor):

    segm: Tensor = segm_net(imgs)
    L, mask = be_sample(segm)

    segment = Transformer.get_random_segment(mask)
    split_gan.train(imgs, segment)


def train_segm(imgs: Tensor):

    segm: Tensor = segm_net(imgs)
    L, mask = be_sample(segm)

    loss = gan.generator_loss(imgs, Mask(segm)) + NeighbourDiffLoss.__call__(segm) * 5 - Loss(L)/2

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
    (cycle_loss / 5).minimize()
    opt.step()
    # gan.optimizer.opt_min.step()

    # segm: Tensor = segm_net(imgs)
    # L, mask = be_sample(segm)
    # segment = Transformer.get_random_segment(mask)
    # split_loss = (split_gan.generator_loss(imgs, segment) + NeighbourDiffLoss.__call__(mask.tensor) * 10) * L
    #
    # opt.zero_grad()
    # (split_loss / 3).minimize()
    # opt.step()


def train_sup_segm(imgs: Tensor, mask: Mask):

    segm: Tensor = segm_net(imgs)

    loss = nn.BCELoss()(segm, mask.tensor)

    opt.zero_grad()
    loss.backward()
    opt.step()


def show_fake_and_segm(imgs: Tensor):

    with torch.no_grad():
        segm: Tensor = segm_net(imgs[:16])
        fake = gan.gan_model.generator(segm).detach()
        show_images(fake.cpu(), 4, 4)
        show_segmentation(segm.cpu())


def show_split_segm(imgs: Tensor):

    with torch.no_grad():
        segm: Tensor = segm_net(imgs[:16])
        segment = Transformer.get_random_segment(Mask(segm))
        front, bk = split_gan.test(imgs[:16], segment)
        show_images(front.cpu(), 4, 4)
        show_images(bk.cpu(), 4, 4)
        show_segmentation(segm.cpu())


print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, (imgs, labels) in enumerate(dataloader, 0):

        imgs = imgs.to(ParallelConfig.MAIN_DEVICE)

        train_gan(imgs)
        # train_split_gan(imgs)
        train_segm(imgs)

        if i % 20 == 0:
            # show_split_segm(imgs)
            show_fake_and_segm(imgs)







