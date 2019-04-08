import torch


def create_meshgrid(size, dtype=torch.float32):
    """Create coordinate grid.

    Args:
        size (tuple): final grid size
        dtype: final grid type

    Returns:
        tensor of shape `(len(size),) + size`
    """
    return torch.stack(torch.meshgrid(*(torch.arange(s, dtype=dtype) for s in size)))


def orthogonal_slice(volume):
    """Orthogonal centered slices of a volume.

    Args:
        volume: 5D tensor

    Returns:
        3-tuple of 4D tensors
    """
    return (
        volume[:, :, volume.size(2) // 2, :, :],
        volume[:, :, :, volume.size(3) // 2, :],
        volume[:, :, :, :, volume.size(4) // 2],
    )


def logits_to_one_hot(tensor, dtype=None):
    """Transform logits or softmax encoded segmentation mask to one-hot.

    Args:
        tensor:
        dtype:

    Returns:

    """
    out = torch.zeros_like(tensor, dtype=dtype)
    out.scatter_(1, tensor.max(1)[1].unsqueeze(1), 1)
    return out
