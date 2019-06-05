# Root directory for dataset
from typing import List

import torch
from torch import nn
from torch.distributions import Bernoulli
from torchvision.datasets import Cityscapes
from torchvision.transforms import transforms

from data_loader.data2d.segmentation_transform import Transformer
from framework.nn.modules.gan.GANModel import ConditionalGANModel
from framework.nn.modules.gan.image2image.gan_factory import MaskToImageFactory
from framework.optim.min_max import MinMaxOptimizer, MinMaxLoss
from framework.nn.modules.gan.penalties.AdaptiveLipschitzPenalty import AdaptiveLipschitzPenalty
from framework.nn.modules.gan.penalties.l2_penalty import L2Penalty
from framework.nn.modules.gan.vgg.gan_loss import VggGeneratorLoss
from framework.nn.modules.gan.wgan.WassersteinLoss import WassersteinLoss
from framework.nn.ops.segmentation.Mask import MaskFactory, Mask
from framework.parallel import ParallelConfig
from framework.segmentation.split_and_fill import SplitAndFill
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

netG, netD = MaskToImageFactory(image_size, nz, ngf, ndf, nc, labels_list)

lrG = 0.0001
lrD = 0.0001

# gan_model = ConditionalGANModel(
#     netG,
#     netD,
#     WassersteinLoss(2)
#         .add_penalty(AdaptiveLipschitzPenalty(1, 0.01))
#         .add_penalty(L2Penalty(1))
# )

# vgg_loss_fn = VggGeneratorLoss(ParallelConfig.MAIN_DEVICE)

# optimizer = MinMaxOptimizer(gan_model.parameters(), lrG, lrD)

# optG = torch.optim.Adam(gan_model.parameters().min_parameters, lr=0.0001, betas=(0.5, 0.9))

split_model = SplitAndFill(image_size)

segm_net = nn.DataParallel(
    UNetSegmentation(labels_list.__len__()).to(ParallelConfig.MAIN_DEVICE),
    ParallelConfig.GPU_IDS
)
segm_opt = torch.optim.Adam(segm_net.parameters(), lr=0.001)

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, (imgs, labels) in enumerate(dataloader, 0):

        if labels.size(0) != batch_size:
            break

        imgs = imgs.to(ParallelConfig.MAIN_DEVICE)
        # mask = MaskFactory.from_class_map(labels.to(ParallelConfig.MAIN_DEVICE), labels_list)

        segm: torch.Tensor = segm_net(imgs)

        dist = Bernoulli(segm)
        sample: torch.Tensor = dist.sample()
        L = (segm * sample + (1 - segm) * (1 - sample)).log().sum() / segm.numel()

        segment = Transformer.get_random_segment(Mask(sample))
        split_model.train(imgs, segment)

        segm: torch.Tensor = segm_net(imgs)

        segment_test = Transformer.get_random_segment(Mask(segm))
        segm_loss = split_model.generator_loss(imgs, segment_test)

        segm_net.zero_grad()
        segm_loss.minimize()
        segm_opt.step()
        # show_images(img_segment.detach().cpu(), 4, 4)
        print(i)

        if i % 20 == 0:
            with torch.no_grad():
                front, bk = split_model.test(imgs, segment)
                show_images(front.detach().cpu(), 4, 4)
                show_images(bk.detach().cpu(), 4, 4)
                show_segmentation(sample)

        # fake = gan_model.generator.forward(mask.data)
        # vgg_loss = vgg_loss_fn.forward(fake, imgs_cuda)

        # optG.zero_grad()
        # vgg_loss.minimize()
        # optG.step()

        # loss: MinMaxLoss = gan_model.loss_pair(imgs, mask.data)
        # optimizer.train_step(loss)

        # loss: MinMaxLoss = gan_model.loss_pair(img_segment, mask_segment.data)
        # optimizer.train_step(loss)

        # loss1 = gan_model.loss_pair(imgs_cuda, mask.data)
        # optimizer.train_step(loss1)

        # Output training stats
        # if i % 1 == 0:
        #     print('[%d/%d][%d/%d]\tD_Loss: %.4f\tG_Loss: %.4f\tVgg_Loss: %.4f'
        #           % (epoch, num_epochs, i, len(dataloader),
        #              loss.max_loss.item(), loss.min_loss.item(), 1))

        # if (i % 20 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
        #     with torch.no_grad():
        #         imlist = netG.forward(mask.data).detach().cpu()
        #         # show_images(mask.data.detach().cpu(), 4, 4)
        #         show_images(imlist, 4, 4)
        #     show_segmentation(mask)






