# Root directory for datasets
from typing import List

import torch
from torch import nn, Tensor
from torch.distributions import Bernoulli
from torchvision.datasets import Cityscapes
from torchvision.transforms import transforms

from framework.segmentation.Mask import MaskFactory
from framework.parallel import ParallelConfig
from framework.segmentation.unet import UNetSegmentation
from viz.visualization import show_images, show_segmentation

# Number of workers for dataloader
workers = 20
# Batch size during training
batch_size = 16
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


dataroot = "/home/nazar/PycharmProjects/segmentation_data"
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

segm_net = UNetSegmentation(labels_list.__len__()).to(ParallelConfig.MAIN_DEVICE)

opt = torch.optim.Adam(segm_net.parameters(), lr=0.001)


def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht or w != wt:
        input = torch.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = torch.cross_entropy(
        input, target, weight=weight, reduction='mean', ignore_index=250
    )
    return loss

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, (imgs, labels) in enumerate(dataloader, 0):

        imgs = imgs.to(ParallelConfig.MAIN_DEVICE)
        labels = labels.to(ParallelConfig.MAIN_DEVICE)

        segm: Tensor = segm_net(imgs)
        mask = MaskFactory.from_class_map(labels, labels_list)

        dist = Bernoulli(segm)
        sample: Tensor = dist.sample()
        L = (segm * sample + (1 - segm) * (1 - sample)).log().sum() / segm.numel()

        print(L.item())
        loss = nn.BCELoss()(segm, mask.tensor) - L
        # loss = cross_entropy2d(segm, labels)

        opt.zero_grad()
        loss.backward()
        opt.step()

        # Output training stats
        if i % 1 == 0:
            print(loss.item())
        #     print('[%d/%d][%d/%d]\tD_Loss: %.4f\tG_Loss: %.4f\tVgg_Loss: %.4f'
        #           % (epoch, num_epochs, i, len(dataloader),
        #              loss.max_loss.item(), loss.min_loss.item(), 1))

        if i % 20 == 0:
            with torch.no_grad():
                show_images(imgs.cpu(), 4, 4)
                show_segmentation(segm.cpu())






