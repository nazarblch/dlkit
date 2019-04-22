from torch import Tensor


class Loss(Tensor):

    def __ipow__(self, other):
        pass

    def __init__(self, other: Tensor):
        super(Loss, self).__init__(other)
