from __future__ import print_function
import random

import torch
from torch import Tensor
import matplotlib.pyplot as plt

from framework.gan.gan_model import GANModel
from framework.gan.euclidean.discriminator import EDiscriminator
from framework.gan.euclidean.generator import EGenerator
from framework.gan.loss.hinge import HingeLoss
from framework.gan.noise.normal import NormalNoise
from framework.optim.min_max import MinMaxOptimizer
from framework.gan.loss.penalties.adaptive_lipschitz import AdaptiveLipschitzPenalty
from framework.gan.loss.wasserstein import WassersteinLoss

batch_size = 256
noise_size = 2


device = torch.device("cuda:1")

noise = NormalNoise(noise_size, device)
netG = EGenerator(noise).to(device)
print(netG)

netD = EDiscriminator().to(device)
print(netD)


lr = 0.001
betas = (0.5, 0.999)

gan_model = GANModel(
    netG,
    netD,
    HingeLoss()
)

optimizer = MinMaxOptimizer(gan_model.parameters(), lr, 2 * lr, betas)

n = 5000

xs = (torch.arange(0, n, dtype=torch.float32) / 100.0).view(n, 1)
ys = torch.cat((xs.cos(), xs.sin()), dim=1)


plt.scatter(ys[:, 0].view(n).numpy(), ys[:, 1].view(n).numpy())


print("Starting Training Loop...")


def gen_batch() -> Tensor:
    i = random.randint(0, n - batch_size)
    j = i + batch_size
    return ys[i:j, :]


for iter in range(0, 3000):

    data = gen_batch().to(device)

    loss = gan_model.loss_pair(data)
    optimizer.train_step(loss)

    if iter % 100 == 0:
        # print(gan_model.loss.get_penalties()[1].weight)
        print(str(loss.max_loss.item()) + ", g = " + str(loss.min_loss.item()))


fake = netG.forward(3 * batch_size)
plt.scatter(fake[:, 0].cpu().view(3 * batch_size).detach().numpy(),
            fake[:, 1].cpu().view(3 * batch_size).detach().numpy())
plt.show()

