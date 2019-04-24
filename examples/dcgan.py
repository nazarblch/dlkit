from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

from framework.nn.modules.gan.dcgan.DCGANModel import DCGANModel
from framework.nn.modules.gan.dcgan.Discriminator import Discriminator
from framework.nn.modules.gan.dcgan.Generator import Generator

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


netG = Generator(batch_size, noise_size, device)
netG.apply(weights_init)
print(netG)

netD = Discriminator(device)
netD.apply(weights_init)
print(netD)


lr = 0.0002
beta1 = 0.5
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.99))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.99))

gan_model = DCGANModel(netD, batch_size, device)

iters = 0

print("Starting Training Loop...")

for epoch in range(5):
    for i, data in enumerate(dataloader, 0):

        imgs = data[0].to(device)

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        fake = netG.gen_and_forward()
        netD.zero_grad()
        errD = gan_model.discriminator_loss(imgs, fake)
        errD.backward()
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################

        netG.zero_grad()
        errG = gan_model.generator_loss(fake)
        errG.backward()
        optimizerG.step()

        # Output training stats
        if i % 10 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                  % (epoch, 5, i, len(dataloader),
                     errD.item(), errG.item()))

        if iters % 50 == 0:
            with torch.no_grad():
                fake = netG.gen_and_forward().detach().cpu()
                plt.figure(figsize=(8, 8))
                plt.axis("off")
                plt.title("Training Images")
                plt.imshow(
                    np.transpose(vutils.make_grid(fake[:64], padding=2, normalize=True).cpu(),
                                 (1, 2, 0)))
                plt.show()

        iters += 1
