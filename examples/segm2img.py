# Root directory for datasets
from typing import List

import torch
from torchvision.datasets import Cityscapes
from torchvision.transforms import transforms

from framework.segmentation.Mask import MaskFactory
from framework.parallel import ParallelConfig
from framework.segmentation.mask_to_image import MaskToImage
from viz.visualization import show_images

# Number of workers for dataloader
workers = 10
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

gan = MaskToImage(image_size, len(labels_list))


print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, (imgs, labels) in enumerate(dataloader, 0):

        if labels.size(0) != batch_size:
            break

        imgs = imgs.to(ParallelConfig.MAIN_DEVICE).tanh()
        mask = MaskFactory.from_class_map(labels.to(ParallelConfig.MAIN_DEVICE), labels_list)

        gan.train(imgs, mask)

        print(i)

        if i % 20 == 0:
            with torch.no_grad():
                imlist = gan.gan_model.generator.forward(mask.tensor).detach().cpu()
                # show_images(mask.data.detach().cpu(), 4, 4)
                show_images(imlist, 4, 4)
                # show_segmentation(mask.data)






