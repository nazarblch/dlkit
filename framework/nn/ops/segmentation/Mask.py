from typing import List

import torch
from torch import Tensor, LongTensor, FloatTensor


class Mask:

    def __init__(self, data: Tensor):
        self.data = data


class MaskFactory:

    @staticmethod
    def get_segment_mask(class_map: LongTensor, index: int) -> Tensor:
        return class_map.eq(index).float()

    @staticmethod
    def from_class_map(labels: LongTensor, labels_list: List[int]) -> Mask:

        masks = [MaskFactory.get_segment_mask(labels, i).view(labels.size(0), 1, *(labels.size()[1:]))
                 for i in labels_list]

        return Mask(torch.cat(masks, dim=1))

