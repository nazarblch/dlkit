import random

import torch
from torch import Tensor


class Transformer:

    @staticmethod
    def get_random_segment_batch(imgs: Tensor, masks: Tensor):
        img_list = []
        mask_list = []
        for i in range(0, imgs.size(0)):
            img_i, mask_i = Transformer.get_random_segment(imgs[i], masks[i])
            img_list.append(img_i)
            mask_list.append(mask_i)

        return torch.cat(img_list, dim=0), torch.cat(mask_list, dim=0)

    @staticmethod
    def get_random_segment(img, mask):

        nc = mask.size(0)

        index = random.randint(0, nc-1)

        sum = mask.sum()
        while mask[index].sum() < 0.05 * sum:
            index = random.randint(0, nc-1)

        mm = torch.zeros([1, nc, 1, 1], device=mask.device, dtype=torch.float32)
        mm[0, index, 0, 0] = 1

        segment = mask[index].view(1, 1, *mask.size()[1:])

        return img * segment, mask * mm
