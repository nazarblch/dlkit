from torch import Tensor


class Loss(Tensor):

    def __ipow__(self, other):
        pass

    def __init__(self, other: Tensor):
        super(Loss, self).__init__(other)

    def __add__(self, other):
        return Loss(super(Loss, self) + other)

    def __mul__(self, weight: float):
        return Loss(super(Loss, self).__mul__(weight))
