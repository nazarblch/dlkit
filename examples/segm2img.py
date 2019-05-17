# Root directory for dataset
from typing import List

import torch
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import transforms

from data_loader.data2d.segmentation_transform import Transformer
from framework.nn.modules.gan.GANModel import ConditionalGANModel
from framework.nn.modules.gan.image2image.gan_factory import MaskToImageFactory, ImageToImageFactory
from framework.nn.modules.gan.optimize import GANOptimizer
from framework.nn.modules.gan.penalties.AdaptiveLipschitzPenalty import AdaptiveLipschitzPenalty
from framework.nn.modules.gan.wgan.WassersteinLoss import WassersteinLoss
from framework.nn.ops.segmentation.Mask import MaskFactory
from viz.visualization import show_images, show_segmentation

# Number of workers for dataloader
workers = 12
# Batch size during training
batch_size = 32
# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 128
# Number of channels in the training images. For color images this is 3
nc = 3
# Size of z latent vector (i.e. size of generator input)
nz = 200
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64
# Number of training epochs
num_epochs = 30


# dataset = CocoStuff10k(
#         root="/home/nazar/PycharmProjects/data",
#         split="all",
#         ignore_label=255,
#         mean_bgr=(104.008, 116.669, 122.675),
#         augment=True,
#         crop_size=64,
#         scales=(0.5, 0.75, 1.0, 1.25, 1.5),
#         flip=True,
#     )
# print(dataset)
dataroot = "/home/nazar/PycharmProjects/data/"
dataset = VOCSegmentation(dataroot,
                        year='2012',
                        # split='train',
                        # mode='fine',
                        # target_type='instance',
                        transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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

# Decide which device we want to run on
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")


G_losses = []
D_losses = []

labels_list: List[int] = list(range(0, 20))

netG, netD = ImageToImageFactory(image_size, nz, ngf, ndf, nc, device)

lrG = 0.0001
lrD = 0.0002

gan_model = ConditionalGANModel(
    netG,
    netD,
    WassersteinLoss(2.5).add_penalty(AdaptiveLipschitzPenalty(1, 0.01))
)

optimizer = GANOptimizer(gan_model.parameters(), lrG, lrD)


print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, (imgs, labels) in enumerate(dataloader, 0):

        if labels.size(0) != batch_size:
            break

        imgs_cuda = imgs.to(device)
        mask = MaskFactory.from_class_map(labels.to(device), labels_list)

        # show_segmentation(mask.data)

        # img_segment, mask_segment = Transformer.get_random_segment_batch(imgs_cuda, mask)
        img_erase = Transformer.erase_random_segment_batch(imgs_cuda, mask.data)
        # print(labels.unique().numpy().tolist())
        # show_images(img_erase.detach().cpu(), 4, 4)
        # show_images(img_erase.detach().cpu(), 4, 4)

        loss = gan_model.loss_pair(imgs_cuda, img_erase)
        optimizer.train_step(loss)

        # loss1 = gan_model.loss_pair(imgs_cuda, mask.data)
        # optimizer.train_step(loss1)

        G_losses.append(loss.generator_loss.item())
        D_losses.append(loss.discriminator_loss.item())

        # Output training stats
        if i % 10 == 0:
            print('[%d/%d][%d/%d]\tD_Loss: %.4f\tG_Loss: %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                        sum(D_losses)/len(D_losses), sum(G_losses)/len(G_losses)))

        if (i % 20 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                imlist = netG.forward(batch_size, img_erase).detach().cpu()
                show_images(img_erase.detach().cpu(), 4, 4)
                show_images(imlist, 4, 4)
        #     show_segmentation(mask)






