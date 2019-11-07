from typing import Callable
from torch import Tensor
from trainer.data_space import DataSpace, SpaceToDataMap


class Mapping:

    def __init__(
            self,
            from_space: DataSpace,
            to_space: DataSpace,
            function: Callable[[SpaceToDataMap], SpaceToDataMap]):

        self.from_space = from_space
        self.to_space = to_space
        self.function = function
