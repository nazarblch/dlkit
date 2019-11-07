import torch

from torch import Tensor


class OneHot:

    """
    Transform logits or softmax encoded segmentation mask to one-hot.
    """
    @staticmethod
    def __call__(logits: Tensor):

        out: Tensor = torch.zeros_like(logits)
        out.scatter_(1, logits.argmax(1).unsqueeze(1), 1)
        return out

