import torch
import torch.nn as nn
import torch.nn.functional as F
from dlkit.nn import functional


__all__ = ('Upsample', 'View', 'AppendGrid', 'Scale')


class AppendGrid(nn.Module):

    def __init__(self):
        super(AppendGrid, self).__init__()
        self.grid = None

    def forward(self, features):
        if self.grid is None:
            if features.dim() == 4:
                size = features.shape[-2:]
            elif features.dim() == 5:
                size = features.shape[-3:]
            else:
                raise ValueError('4D or 5D input tensor is expected')
            self.grid = functional.create_meshgrid(size, features.dtype).to(features.device)
            self.grid.unsqueeze_(0)
        grid = self.grid.expand(features.size(0), *self.grid.size()[1:])
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
        return F.interpolate(inputs, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                             align_corners=False)

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
