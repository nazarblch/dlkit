from typing import Optional

import torch
from torch import Tensor


class Loss:

    def __init__(self, tensor: Tensor):
        self.__tensor = tensor

    def __add__(self, other):
        return Loss(self.__tensor + other.to_tensor())

    def __sub__(self, other):
        return Loss(self.__tensor - other.to_tensor())

    def __mul__(self, weight: float):
        return Loss(self.__tensor * weight)

    def __truediv__(self, weight: float):
        return Loss(self.__tensor / weight)

    def minimize(self) -> None:
        return self.__tensor.backward()

    def maximize(self) -> None:
        return self.__tensor.backward(-torch.ones_like(self.__tensor))

    def item(self) -> float:
        return self.__tensor.item()

    def to_tensor(self) -> Tensor:
        return self.__tensor
