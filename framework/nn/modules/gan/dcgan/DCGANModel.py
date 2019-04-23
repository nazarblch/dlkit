import torch
from torch import Tensor, nn

from framework.Loss import Loss
from framework.nn.modules.gan.Discriminator import Discriminator
from framework.nn.modules.gan.GANLoss import GANLoss


class DCGANModel(GANLoss):

    __criterion = nn.BCELoss()

    def __init__(self, discriminator: Discriminator, batch_size: int, device: torch.device):
        super().__init__(discriminator)
        self.batch_size = batch_size
        self.device = device

    def _generator_loss(self, Gz: Tensor) -> Loss:
        real_labels = torch.full((self.batch_size,), 1, device=self.device)
        errG = self.__criterion(self.discriminator(Gz).view(-1), real_labels)
        return errG

    def _discriminator_loss(self, x: Tensor, y: Tensor) -> Loss:

        real_labels = torch.full((self.batch_size,), 1, device=self.device)
        err_real = self.__criterion(x.view(-1), real_labels)

        fake_labels = torch.full((self.batch_size,), 0, device=self.device)
        err_fake = self.__criterion(y.view(-1), fake_labels)

        return err_fake + err_real

