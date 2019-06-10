import torch
from torch import Tensor, nn

from framework.Loss import Loss
from framework.parallel import ParallelConfig


class NeighbourDiffLoss:

    neighbour_filter: Tensor = torch.zeros(4, 1, 3, 3, dtype=torch.float32).to(ParallelConfig.MAIN_DEVICE)
    neighbour_filter[:, 0, 1, 1] = 1
    neighbour_filter[0, 0, 0, 1] = -1
    neighbour_filter[1, 0, 1, 0] = -1
    neighbour_filter[2, 0, 1, 2] = -1
    neighbour_filter[3, 0, 2, 1] = -1

    @staticmethod
    def __call__(segm: Tensor) -> Loss:

        res = 0

        for segm_i in segm.split(1, dim=1):
            res += nn.functional.conv2d(segm_i, NeighbourDiffLoss.neighbour_filter).abs().mean()

        return Loss(res / segm.shape[1])
