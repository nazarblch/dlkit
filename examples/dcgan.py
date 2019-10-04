from __future__ import print_function
#%matplotlib inline
import random
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.nn import init

from data.path import DataPath
from framework.gan.GANModel import GANModel
from framework.gan.dcgan.discriminator import DCDiscriminator
from framework.gan.dcgan.generator import DCGenerator
from framework.gan.dcgan.model import DCGANLoss
from framework.gan.dcgan.vgg_discriminator import VGGDiscriminator
from framework.gan.loss.hinge import HingeLoss
from framework.gan.noise.normal import NormalNoise
from framework.optim.min_max import MinMaxOptimizer
from viz.visualization import show_images

manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)
print(torch.cuda.is_available())

batch_size = 32
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

device = torch.device("cuda:0")


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
netG = DCGenerator(noise, image_size).to(device)
netG.apply(weights_init)
print(netG)

netD = DCDiscriminator().to(device)
netD.apply(weights_init)
print(netD)

netDV = VGGDiscriminator().to(device)
netDV.main.apply(weights_init)
print(netDV)


lr = 0.0002
betas = (0.5, 0.999)

gan_model = GANModel(netG, netD, HingeLoss())
optimizer = MinMaxOptimizer(gan_model.parameters(), lr, lr * 2)

gan_model_v = GANModel(netG, netDV, HingeLoss())
optimizer_v = MinMaxOptimizer(gan_model.parameters(), lr, lr * 2)

iters = 0

print("Starting Training Loop...")

for epoch in range(5):
    for i, data in enumerate(dataloader, 0):

        imgs = data[0].to(device)

        # loss = gan_model_v.loss_pair(imgs)
        # optimizer_v.train_step(loss)

        loss = gan_model.loss_pair(imgs)
        optimizer.train_step(loss)

        # Output training stats
        if i % 10 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                  % (epoch, 5, i, len(dataloader),
                     loss.max_loss.item(), loss.min_loss.item()))

        if iters % 50 == 0:
            with torch.no_grad():
                fake = netG.forward(batch_size).detach().cpu()
                show_images(fake, 4, 4)

        iters += 1
