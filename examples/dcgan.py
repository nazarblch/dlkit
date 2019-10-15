from __future__ import print_function
#%matplotlib inline
import random
import time

import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.nn import init
from torch import nn, Tensor
from data.path import DataPath
from framework.Loss import Loss
from framework.gan.dcgan.encoder import DCEncoder
from framework.gan.gan_model import GANModel
from framework.gan.cycle.model import CycleGAN
from framework.gan.dcgan.discriminator import DCDiscriminator, ResDCDiscriminator
from framework.gan.dcgan.generator import DCGenerator, ResDCGenerator
from framework.gan.dcgan.model import DCGANLoss
from framework.gan.dcgan.vgg_discriminator import VGGDiscriminator
from framework.gan.euclidean.discriminator import EDiscriminator
from framework.gan.loss.hinge import HingeLoss
from framework.gan.loss.penalties.lipschitz import ApproxLipschitzPenalty, LipschitzPenalty
from framework.gan.loss.wasserstein import WassersteinLoss
from framework.gan.noise.normal import NormalNoise
from framework.gan.vgg.gan_loss import VggGeneratorLoss
from framework.optim.min_max import MinMaxOptimizer
from framework.parallel import ParallelConfig
from viz.visualization import show_images

manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)
print(torch.cuda.is_available())

batch_size = 64
image_size = 128
noise_size = 100

dataset = dset.ImageFolder(root=DataPath.CelebA.HOME,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=12)

device = torch.device("cuda:1")

noise = NormalNoise(noise_size, device)
netG = ResDCGenerator(noise, image_size).to(device)
print(netG)

netD = ResDCDiscriminator().to(device)
print(netD)


lr = 0.0002
betas = (0.5, 0.999)

gan_model = GANModel(netG, HingeLoss(netD))

netG_back = DCEncoder(nc_out=noise_size).to(device)
print(netG_back)

netD_z = EDiscriminator(dim=noise_size, ndf=100).to(device)
print(netD_z)


gan_model_back = GANModel(netG_back, HingeLoss(netD_z))


l1_loss = nn.L1Loss()

cycle_gan = CycleGAN[Tensor, Tensor](
    netG,
    netG_back,
    loss_1=lambda z1, z2: Loss(l1_loss(z1, z2)),
    loss_2=lambda img1, img2: Loss(l1_loss(img1, img2)),
    lr=0.0002
)

iters = 0

print("Starting Training Loop...")
t0 = time.time()

for epoch in range(5):
    for i, data in enumerate(dataloader, 0):

        imgs = data[0].to(device)
        z = noise.sample(batch_size)

        loss = gan_model.train(imgs, z)

        loss_back = gan_model_back.train(z, imgs)

        cycle_gan.train(z, imgs)

        # Output training stats
        if i % 10 == 0:
            print(time.time() - t0)
            t0 = time.time()
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                  % (epoch, 5, i, len(dataloader),
                     loss[1], loss[0]))

        if iters % 50 == 0:
            with torch.no_grad():
                fake = netG.forward(z).detach().cpu()
                show_images(fake, 4, 4)

        iters += 1
