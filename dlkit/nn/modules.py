import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ('Upsample', 'View', 'AppendGrid2d', 'AppendGrid3d', 'upsample')


class AppendGrid3d(nn.Module):

    def __init__(self, size, dtype=torch.float32, device='cuda'):
        super(AppendGrid3d, self).__init__()
        self.size = size
        self.grid = create_3d_meshgrid(size, dtype, device)

    def forward(self, features):
        grid = self.grid.unsqueeze(0).expand(features.size(0), *self.grid.size())
        return torch.cat((features, grid), dim=1)


class AppendGrid2d(nn.Module):

    def __init__(self, size, dtype=torch.float32, device='cuda'):
        super(AppendGrid2d, self).__init__()
        self.size = size
        self.grid = create_2d_meshgrid(size, dtype, device)

    def forward(self, features):
        grid = self.grid.unsqueeze(0).expand(features.size(0), *self.grid.size())
        return torch.cat((features, grid), dim=1)


class Scale(nn.Module):

    def __init__(self):
        super(Scale, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, dtype=torch.float32))

    def forward(self, input):
        return input * self.weight


class Upsample(nn.Module):

    def __init__(self, size=None, scale_factor=None, mode='bilinear'):
        super(Upsample, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, inputs):
        return F.interpolate(inputs, size=size, scale_factor=scale_factor, mode=mode, align_corners=False)

    def extra_repr(self):
        params = []
        if self.size is not None:
            params.append('size={}'.format(self.size))
        if self.scale_factor is not None:
            params.append('scale_factor={}'.format(self.scale_factor))
        params.append('mode={}'.format(self.mode))
        return ', '.join(params)


class View(nn.Module):

    def __init__(self, *dims):
        super(View, self).__init__()
        self.dims = dims

    def forward(self, inputs):
        return inputs.view(*self.dims)

    def extra_repr(self):
        return ', '.join(str(dim) for dim in self.dims)
