from abc import ABC, abstractmethod
from typing import Callable, List

from torch import Tensor

from dlkit.Loss import Loss


class DiscriminatorPenalty(ABC):

    @abstractmethod
    def __call__(self,
                 discriminator: Callable[[List[Tensor]], Tensor],
                 x: List[Tensor]) -> Loss: pass