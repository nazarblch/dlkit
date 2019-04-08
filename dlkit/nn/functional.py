import torch


def create_3d_meshgrid(size, dtype=torch.float32, device='cpu'):
    if isinstance(size, int):
        size = (size, size, size)
    assert len(size) == 3
    x = torch.stack(
        torch.meshgrid(
            torch.arange(size[0]),
            torch.arange(size[1]),
            torch.arange(size[2]),
        )
    )
    return x.type(dtype).to(device)


def create_2d_meshgrid(size, dtype=torch.float32, device='cpu'):
    if isinstance(size, int):
        size = (size, size)
    assert len(size) == 2
    x = torch.stack(
        torch.meshgrid(
            torch.arange(size[0]),
            torch.arange(size[1]),
        )
    )
    return x.type(dtype).to(device)
