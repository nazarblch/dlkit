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
from framework.gan.GANModel import GANModel
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


# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         nn.init.normal_(m.weight.data, 1.0, 0.02)
#         nn.init.constant_(m.bias.data, 0)

def weights_init(net, init_type='normal', gain=0.02):
    """Get different initial method for the network weights"""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv')!=-1 or classname.find('Linear')!=-1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


noise = NormalNoise(noise_size, device)
netG = nn.DataParallel(ResDCGenerator(noise, image_size).to(device), ParallelConfig.GPU_IDS)
netG.apply(weights_init)
print(netG)

netD = nn.DataParallel(ResDCDiscriminator().to(device), ParallelConfig.GPU_IDS)
netD.apply(weights_init)
print(netD)

# netDV = VGGDiscriminator().to(device)
# netDV.main.apply(weights_init)
# netDV = nn.DataParallel(netDV, ParallelConfig.GPU_IDS)
# print(netDV)


lr = 0.0002
betas = (0.5, 0.999)

gan_model = GANModel(netG, netD, HingeLoss())
optimizer = MinMaxOptimizer(gan_model.parameters(), lr, lr * 2)


netG_back = nn.DataParallel(DCDiscriminator(nc_out=noise_size).to(device), ParallelConfig.GPU_IDS)
netG_back.apply(weights_init)
print(netG_back)

netD_z = nn.DataParallel(EDiscriminator(dim=noise_size, ndf=100).to(device), ParallelConfig.GPU_IDS)
netD_z.apply(weights_init)
print(netD_z)


gan_model_back = GANModel(netG_back, netD_z, HingeLoss())
optimizer_back = MinMaxOptimizer(gan_model_back.parameters(), lr, lr * 2)

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

        loss = gan_model.loss_pair(imgs, z)
        optimizer.train_step(loss)

        loss_back = gan_model_back.loss_pair(z, imgs)
        optimizer_back.train_step(loss_back)

        cycle_gan.train(z, imgs)

        # Output training stats
        if i % 10 == 0:
            print(time.time() - t0)
            t0 = time.time()
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                  % (epoch, 5, i, len(dataloader),
                     loss.max_loss.item(), loss.min_loss.item()))

        if iters % 50 == 0:
            with torch.no_grad():
                fake = netG.forward(z).detach().cpu()
                show_images(fake, 4, 4)

        iters += 1
