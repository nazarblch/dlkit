# Root directory for datasets
from typing import List, Tuple
import matplotlib.pyplot as plt
import albumentations
import cv2
import torch
from torch import nn, Tensor
from torch.distributions import Bernoulli, Categorical
from torchvision.datasets import Cityscapes
from torchvision.transforms import transforms
import numpy as np
from data.d2.datasets.superpixels import Superpixels
from data.path import DataPath
from framework.Loss import Loss
from framework.gan.cycle.model import CycleGAN
from framework.module import NamedModule
from framework.segmentation.Mask import MaskFactory, Mask
from framework.parallel import ParallelConfig
from framework.segmentation.base import PenalizedSegmentation
from framework.segmentation.fcn import FCNSegmentation
from framework.segmentation.loss.modularity import VGGModularity
from framework.segmentation.loss.sp_loss import SuperPixelsLoss
from framework.segmentation.mask_to_image import MaskToImage
from framework.segmentation.unet import UNetSegmentation
from viz.visualization import show_segmentation, show_images, show_image

# Number of workers for dataloader
workers = 10
# Batch size during training
batch_size = 32
# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 256
# Number of training epochs
num_epochs = 300
labels_count = 40

dataset = Superpixels(
    root=DataPath.CatsAndDogs.HOME_TRAIN + "/cats",
    target_root="/home/nazar/Downloads/dogvscat/sp/dataset/training_set/cats",
    compute_sp=False,
    transforms_al=albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.CenterCrop(image_size, image_size),
    ]),
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    target_transform=transforms.ToTensor()
)

dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=workers)


segm_net = PenalizedSegmentation(
    # FCNSegmentation(labels_count, n_conv=3)
    UNetSegmentation(labels_count)
)
segm_net.add_penalty(SuperPixelsLoss(0.01))
# segm_net.add_penalty(VGGModularity(3, 0.001))


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


mask2image = MaskToImage(image_size, labels_count)

l1_loss = nn.L1Loss()
ent_loss = nn.BCELoss()
cycle_gan = CycleGAN(
    NamedModule(mask2image.gan_model.generator, ["mask"], ["image"]),
    NamedModule(segm_net.segmentation, ["image"], ["mask"]),
    loss_1={
        "mask": lambda mask1, mask2: Loss((mask1.sigmoid() - mask2.sigmoid()).abs().mean()),
    },
    loss_2={
        "image": lambda img1, img2: Loss(l1_loss(img1, img2))
    },
    lr=0.0002
)

label_colours = np.random.randint(255, size=(100, 3))
print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, (imgs, sp) in enumerate(dataloader, 0):

        imgs = imgs.to(ParallelConfig.MAIN_DEVICE).type(torch.float32)
        sp = sp.to(ParallelConfig.MAIN_DEVICE).type(torch.int64) + 1
        mask = segm_net.forward(imgs, sp)

        segm_net.train(imgs, sp)
        mask2image.train(imgs, mask.detach())
        loss = mask2image.generator_loss(imgs, mask)
        loss.minimize_step(segm_net.opt)
        cycle_gan.train({"mask": mask.detach()}, {"image": imgs})

        if i % 5 == 0:
            im_target = mask.max(dim=1)[1].cpu().numpy()[0]
            im_target_rgb = np.array([label_colours[c % 100] for c in im_target])
            im_target_rgb = im_target_rgb.reshape((image_size, image_size, 3)).astype(np.uint8)
            plt.imshow(im_target_rgb)
            plt.show()
            show_image(mask2image.forward(mask).detach().cpu()[0])








