import random
from typing import Tuple

import torch
from torch import Tensor

from framework.nn.ops.segmentation.Mask import Mask


class Transformer:

    @staticmethod
    def get_random_segment_batch(images: Tensor, masks: Mask) -> Tuple[Tensor, Mask]:
        img_list = []
        mask_list = []
        for i in range(0, images.size(0)):
            img_i, mask_i = Transformer.get_random_segment(images[i], masks.data[i])
            img_list.append(img_i.view(1, *img_i.size()))
            mask_list.append(mask_i.view(1, *mask_i.size()))

        return torch.cat(img_list, dim=0), Mask(torch.cat(mask_list, dim=0))

    @staticmethod
    def erase_random_segment_batch(images: Tensor, masks: Mask) -> Tensor:
        img_list = []
        for i in range(0, images.size(0)):
            img_i = Transformer.erase_random_segment(images[i], masks.data[i])
            img_list.append(img_i.view(1, *img_i.size()))

        return torch.cat(img_list, dim=0)

    @staticmethod
    def generate_segment_index(mask: Tensor) -> int:

        nc = mask.size(0)

        index = random.randint(0, nc - 1)

        mask_sum = mask.sum()
        i = 0
        while mask[index].sum() < 0.1 * mask_sum and i < nc:
            index = random.randint(0, nc - 1)
            i += 1

        return index

    @staticmethod
    def get_random_segment(img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:

        nc = mask.size(0)

        index = Transformer.generate_segment_index(mask)

        mm = torch.zeros([nc, 1, 1], device=mask.device, dtype=torch.float32)
        mm[index, 0, 0] = 1

        segment = mask[index].view(1, *mask.size()[1:])

        return img * segment, mask * mm

    @staticmethod
    def erase_random_segment(img: Tensor, mask: Tensor) -> Tensor:

        index = Transformer.generate_segment_index(mask)
        segment = mask[index].view(1, *mask.size()[1:])

        return img * (torch.ones_like(segment) - segment)
