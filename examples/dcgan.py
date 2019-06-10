from __future__ import print_function
#%matplotlib inline
import random
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

from framework.gan.GANModel import GANModel
from framework.gan.dcgan.Discriminator import Discriminator
from framework.gan.dcgan import Generator
from framework.gan.noise.normal import NormalNoise
from framework.optim.min_max import MinMaxOptimizer
from framework.gan.penalties.AdaptiveLipschitzPenalty import AdaptiveLipschitzPenalty
from framework.gan.wgan.WassersteinLoss import WassersteinLoss

manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)

dataroot = "/home/nazar/Downloads/celeba"

batch_size = 128
image_size = 64
noise_size = 100

dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=12)

device = torch.device("cuda")


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


noise = NormalNoise(noise_size, device)
netG = Generator(noise).to(device)
netG.apply(weights_init)
print(netG)

netD = Discriminator().to(device)
netD.apply(weights_init)
print(netD)


lr = 0.0001
betas = (0.5, 0.9)

gan_model = GANModel(netG, netD, WassersteinLoss(1).add_penalty(AdaptiveLipschitzPenalty(1, 0.05)))
optimizer = MinMaxOptimizer(gan_model.parameters(), lr, betas)

iters = 0

print("Starting Training Loop...")

for epoch in range(5):
    for i, data in enumerate(dataloader, 0):

        imgs = data[0].to(device)

        loss = gan_model.loss_pair(imgs)
        optimizer.train_step(loss)


        # Output training stats
        if i % 10 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                  % (epoch, 5, i, len(dataloader),
                     loss.discriminator_loss.item(), loss.generator_loss.item()))

        if iters % 50 == 0:
            with torch.no_grad():
                fake = netG.forward(batch_size).detach().cpu()
                plt.figure(figsize=(8, 8))
                plt.axis("off")
                plt.title("Training Images")
                plt.imshow(
                    np.transpose(vutils.make_grid(fake[:64], padding=2, normalize=True).cpu(),
                                 (1, 2, 0)))
                plt.show()

        iters += 1
