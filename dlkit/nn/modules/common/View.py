from typing import List, Iterator, overload

from torch import nn, Tensor


class View(nn.Module):

    @overload
    def __init__(self, *dims: Iterator[int]):
        super(View, self).__init__()
        self.dims = dims

    def __init__(self, other: Tensor):
        super(View, self).__init__()
        self.dims = other.size()

    def forward(self, inputs: Tensor):
        return inputs.view(*self.dims)

    def extra_repr(self):
        return ', '.join(str(dim) for dim in self.dims)



