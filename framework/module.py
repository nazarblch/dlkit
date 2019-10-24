from abc import ABC
from typing import Dict, Generic, TypeVar, List
import torch
from torch import nn, Tensor


class Module(nn.Module, ABC):

    def freeze(module):
        for param in module.parameters():
            param.requires_grad = False

    def unfreeze(module):
        for param in module.parameters():
            param.requires_grad = True

    def num_params(module):
        return sum(p.numel() for p in module.parameters())


class NamedModule(Module):

    def __init__(self,
                 module: nn.Module,
                 from_names: List[str],
                 to_names: List[str]):
        super().__init__()
        self.module = module
        self.from_names = from_names
        self.to_names = to_names

    def forward(self, name2tensor: Dict[str, Tensor]) -> Dict[str, Tensor]:

        assert name2tensor.keys() == set(self.from_names)
        input = [name2tensor[name] for name in self.from_names]

        res = self.module(*input)
        if isinstance(res, Tensor):
            res = [res]

        return dict(zip(self.to_names, res))

    def __call__(self, name2tensor: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return self.forward(name2tensor)
