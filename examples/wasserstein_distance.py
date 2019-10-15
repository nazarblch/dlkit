import torch
from torch import optim

from framework.gan.gan_model import GANModel
from framework.gan.euclidean import discriminator
from framework.gan.loss.penalties.adaptive_lipschitz import AdaptiveLipschitzPenalty
from framework.gan.loss.wasserstein import WassersteinLoss

n = 1000
xs = (torch.arange(0, n, dtype=torch.float32) / 100.0).view(n, 1)
device = torch.device("cpu")

ys1: torch.Tensor = torch.cat((xs.cos(), xs.sin()), dim=1).to(device)
ys2: torch.Tensor = torch.cat((xs.cos(), xs.sin()), dim=1).to(device) * 5

netD = discriminator().to(device)
print(netD)

gan_model = GANModel(None,
                     netD,
                     WassersteinLoss(0.5).add_penalty(AdaptiveLipschitzPenalty(1, 0.05))
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
