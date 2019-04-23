import torch
import torch.functional as F
from torch import nn


class AppendGrid(nn.Module):

    def __init__(self):
        super(AppendGrid, self).__init__()
        self.grid = None

    @staticmethod
    def create_meshgrid(size, dtype=torch.float32):
        """Create coordinate grid.

        Args:
            size (tuple): final grid size
            dtype: final grid type

        Returns:
            tensor of shape `(len(size),) + size`
        """
        return torch.stack(torch.meshgrid(*(torch.arange(s, dtype=dtype) for s in size)))

    def forward(self, features):
        if self.grid is None:
            if features.dim() == 4:
                size = features.shape[-2:]
            elif features.dim() == 5:
                size = features.shape[-3:]
            else:
                raise ValueError('4D or 5D input tensor is expected')
            self.grid = AppendGrid.create_meshgrid(size, features.dtype).to(features.device)
            self.grid.unsqueeze_(0)
        grid = self.grid.expand(features.size(0), *self.grid.size()[1:])
        return torch.cat((features, grid), dim=1)