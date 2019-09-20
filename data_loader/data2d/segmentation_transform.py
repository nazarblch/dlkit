import random
from typing import Tuple

import torch
from torch import Tensor

from framework.segmentation.Mask import Mask


class Transformer:

    active_ids = set()
    MIN_SEGMENT_FRACTION = 0.05

    @staticmethod
    def generate_segment_index(mask: Tensor) -> int:

        nc = mask.size(0)

        index = random.randint(0, nc - 1)

        mask_sum = mask.sum()
        i = 0
        while mask[index].sum() < mask_sum * Transformer.MIN_SEGMENT_FRACTION and i < nc:
            index = random.randint(0, nc - 1)
            i += 1

        Transformer.active_ids.add(index)

        return index

    @staticmethod
    def get_random_segment(masks: Mask) -> Mask:

        nc = masks.tensor.size(1)
        batch_size = masks.tensor.size(0)
        device = masks.tensor.device
        mm: Tensor = torch.zeros((batch_size, nc, masks.tensor.size(2), masks.tensor.size(3)), device=device, dtype=torch.float32)

        for i in range(batch_size):

            index = Transformer.generate_segment_index(masks.tensor[i])
            mm[i, index, :, :] = 1

        new_mask = masks.tensor[mm == 1].view(batch_size, 1, masks.tensor.size(2), masks.tensor.size(3))

        return Mask(new_mask)

    @staticmethod
    def split_by_random_segment(imgs: Tensor, masks: Mask) -> Tuple[Tensor, Tensor]:

        segment: Tensor = Transformer.get_random_segment(masks).tensor

        return imgs * segment,  imgs * (1 - segment)
