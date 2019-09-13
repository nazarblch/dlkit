from multiprocessing import Pool
from torch import Tensor
import numpy as np
import torch

from framework.Loss import LossModule, Loss
from framework.nn.modules.common.sp_pool import SPPoolMean
from superpixels.mbs import superpixels


class SuperPixelsLoss(LossModule):

    def __init__(self):
        self.pooling = SPPoolMean()
        self.loss = torch.nn.CrossEntropyLoss()
        self.proc = Pool(8)

    def find_sp(self, image: Tensor) -> Tensor:
        if image.is_cuda:
            image = image.cpu()
        img_seg = np.split(image.permute(0, 2, 3, 1).numpy(), image.shape[0], axis=0)

        sp_seq = [superpixels(img) for img in img_seg]
        sp_seq = np.concatenate(sp_seq, axis=0)
        sp_seq = torch.from_numpy(sp_seq).type(torch.int64)

        return sp_seq

    def forward(self, image: Tensor, segm: Tensor) -> Loss:
        sp = self.find_sp(image).to(segm.device)

        print(sp.min(), sp.max())

        sp_argmax = self.pooling.forward(
            segm,
            sp
        ).detach().max(dim=1)[1]

        return Loss(self.loss(segm, sp_argmax))
