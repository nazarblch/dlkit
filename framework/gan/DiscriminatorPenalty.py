from abc import ABC, abstractmethod
from typing import Callable, List

from torch import Tensor

from framework.Loss import Loss


class DiscriminatorPenalty(ABC):

    @abstractmethod
    def __call__(self, dx: Tensor, x: List[Tensor]) -> Loss: pass
