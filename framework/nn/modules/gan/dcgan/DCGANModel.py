import torch
from torch import Tensor, nn

from framework.Loss import Loss
from framework.nn.modules.gan.GANLoss import GANLoss


class DCGANLoss(GANLoss):

    __criterion = nn.BCELoss()

    def generator_loss(self, dgz: Tensor) -> Loss:
        real_labels = torch.full((dgz.size(0),), 1, device=dgz.device)
        errG = self.__criterion(dgz.view(-1).sigmoid(), real_labels)
        return errG

    def discriminator_loss(self, dx: Tensor, dy: Tensor) -> Loss:

        batch_size = dx.size(0)

        real_labels = torch.full((batch_size,), 1, device=dx.device)
        err_real = self.__criterion(dx.view(-1).sigmoid(), real_labels)

        fake_labels = torch.full((batch_size,), 0, device=dx.device)
        err_fake = self.__criterion(dy.view(-1).sigmoid(), fake_labels)

        return -(err_fake + err_real)

