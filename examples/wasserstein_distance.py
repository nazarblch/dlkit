import torch
from torch import optim

from framework.nn.modules.gan.GANModel import GANModel
from framework.nn.modules.gan.euclidean.Discriminator import Discriminator
from framework.nn.modules.gan.penalties.AdaptiveLipschitzPenalty import AdaptiveLipschitzPenalty
from framework.nn.modules.gan.wgan.WassersteinLoss import WassersteinLoss

n = 1000
xs = (torch.arange(0, n, dtype=torch.float32) / 100.0).view(n, 1)
device = torch.device("cuda:0")

ys1: torch.Tensor = torch.cat((xs.cos(), xs.sin()), dim=1).to(device)
ys2: torch.Tensor = torch.cat((xs.cos(), xs.sin()), dim=1).to(device) * 5

netD = Discriminator().to(device)
print(netD)

gan_model = GANModel(None,
                     netD,
                     WassersteinLoss(1).add_penalty(AdaptiveLipschitzPenalty(1, 0.05))
                     )

opt = optim.Adam(netD.parameters(), 0.003, betas=(0.5, 0.9))
opt_slow = optim.Adam(netD.parameters(), 0.0002, betas=(0.5, 0.9))

for iter in range(0, 200):

    opt.zero_grad()
    errD = gan_model.discriminator_loss(ys1, ys2)
    errD.maximize()
    opt.step()

    if iter % 10 == 0:
        w1 = (netD(ys1).mean() - netD(ys2).mean()) / 2
        print(w1.item())


print("slow opt")

for iter in range(0, 200):

    opt.zero_grad()
    errD = gan_model.discriminator_loss(ys1, ys2)
    errD.maximize()
    opt_slow.step()

    if iter % 10 == 0:
        w1 = (netD(ys1).mean() - netD(ys2).mean()) / 2
        print(w1.item())
