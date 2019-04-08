import numpy as np
import torch


def segmap(labels, classes, void=(255, 255, 255)):
    labels = np.asarray(labels)
    img = np.empty(labels.shape + (3,))
    img[:] = void
    for i, _, color in classes:
        img[labels == i] = color
    img /= 255
    return img


def mask_to_rgb(mask, colors):
    if mask.dim() == 4:
        size = mask.shape[-2:]
    elif mask.dim() == 5:
        size = mask.shape[-3:]
    else:
        raise ValueError('4D or 5D input tensor is expected')
    last_dim_index = len(size) + 1
    return (colors[mask.data.max(1)[1].flatten()]
            .view(*((mask.size(0),) + size + (3,)))
            .permute(0, last_dim_index, *range(1, last_dim_index)))


def get_color_table(dataset, device='cpu', dtype=torch.float32):
    colors = [color for _, _, color in dataset.classes]
    return torch.tensor(colors, device=device, dtype=dtype) / 255
