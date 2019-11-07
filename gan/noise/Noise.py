from abc import ABC, abstractmethod

from torch import Tensor


class Noise(ABC):
    @abstractmethod
    def sample(self, n: int) -> Tensor: pass

    @abstractmethod
    def size(self) -> int: pass
