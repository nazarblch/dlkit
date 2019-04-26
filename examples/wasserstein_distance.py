import torch
from torch import optim

from framework.nn.modules.gan.GANModel import GANModel
from framework.nn.modules.gan.euclidian.Discriminator import Discriminator
from framework.nn.modules.gan.penalties.AdaptiveLipschitzPenalty import AdaptiveLipschitzPenalty
from framework.nn.modules.gan.wgan.WassersteinLoss import WassersteinLoss

n = 1000
xs = (torch.arange(0, n, dtype=torch.float32) / 100.0).view(n, 1)
device = torch.device("cuda:1")

ys1: torch.Tensor = torch.cat((xs.cos(), xs.sin()), dim=1).to(device)
ys2: torch.Tensor = torch.cat((xs.cos(), xs.sin()), dim=1).to(device) * 5

netD = Discriminator().to(device)
print(netD)

gan_model = GANModel(None,
                     netD,
                     WassersteinLoss(5).add_penalty(AdaptiveLipschitzPenalty(1, 0.01))
                     )

opt = optim.Adam(netD.parameters(), 0.0003, betas=(0.5, 0.9))

for iter in range(0, 3000):

    opt.zero_grad()
    errD = gan_model.discriminator_loss(ys1, ys2)
    (-errD).backward()
    opt.step()

    if iter % 10 == 0:
        w1 = (netD(ys1).mean() - netD(ys2).mean()) / 2
        print(w1.item())
