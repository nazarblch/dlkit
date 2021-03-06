# Root directory for datasets
from typing import List
import torch
from torch import Tensor, nn
from torchvision.datasets import Cityscapes
from torchvision.transforms import transforms

from data.path import DataPath
from framework.loss.loss import Loss
from gan.cycle import CycleGAN
from gan.noise.normal import NormalNoise
from framework.module import NamedModule
from segmentation.Mask import MaskFactory
from config import ParallelConfig
from segmentation.mask_to_image import MaskToImage
from segmentation.unet import UNetSegmentation
from viz.visualization import show_images

# Number of workers for dataloader
workers = 10
# Batch size during training
batch_size = 64
# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 128
# Number of channels in the training images. For color images this is 3
nc = 3
# Size of z latent vector (i.e. size of generator input)
nz = 10
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64
# Number of training epochs
num_epochs = 30

dataset = Cityscapes(DataPath.CityScapes.HOME,
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

noise=NormalNoise(nz, ParallelConfig.MAIN_DEVICE)

mask2image = MaskToImage(image_size, len(labels_list))


class ImageToMask(nn.Module):

    def forward(self, image: Tensor) -> Tensor:
        return self.segm(image)

    def __init__(self):
        super(ImageToMask, self).__init__()
        self.segm = UNetSegmentation(len(labels_list)).to(ParallelConfig.MAIN_DEVICE)
        # self.img2noize = DCEncoder(nc_out=noise.size()).to(ParallelConfig.MAIN_DEVICE)


image2mask = ImageToMask()

l1_loss = nn.L1Loss()
ent_loss = nn.BCELoss()
cycle_gan = CycleGAN(
    NamedModule(mask2image.gan_model.generator, ["mask"], ["image"]),
    NamedModule(image2mask, ["image"], ["mask"]),
    loss_1={
        "mask": lambda mask1, mask2: Loss(ent_loss(mask1, mask2)),
    },
    loss_2={
        "image": lambda img1, img2: Loss(l1_loss(img1, img2))
    },
    lr=0.0002
)

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, (imgs, labels) in enumerate(dataloader, 0):

        if labels.size(0) != batch_size:
            break

        imgs = imgs.to(ParallelConfig.MAIN_DEVICE)
        mask = MaskFactory.from_class_map(labels.to(ParallelConfig.MAIN_DEVICE), labels_list).tensor
        # z = noise.sample(batch_size)

        mask2image.train(imgs, mask)
        cycle_gan.train({"mask": mask}, {"image": imgs})

        print(i)

        if i % 20 == 0:
            with torch.no_grad():
                imlist = mask2image.forward(mask).detach().cpu()
                # show_images(mask.data.detach().cpu(), 4, 4)
                show_images(imlist, 4, 4)
                # show_segmentation(mask.data)






