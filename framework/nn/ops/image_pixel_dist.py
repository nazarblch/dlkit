from framework.nn.ops.pairwise_map import LocalPairwiseMap2D
from torch import Tensor


class LocalPixelDistance:

    def dist(self, di: int, dj: int, t1: Tensor, t2: Tensor) -> Tensor:
        dt: Tensor = (t2 - t1)
        t_sq = dt.pow(2).mean(dim=2, keepdim=True)
        # print(t_sq.shape, t_sq.mean())
        r2 = di ** 2 + dj ** 2
        return (
          -t_sq / self.sigma_t
          -r2 / self.sigma_i
        ).exp()

    def __init__(self, kernel_size: int, sigma_t: float = 0.1, sigma_i: float = 10):
        self.sigma_t = sigma_t
        self.sigma_i = sigma_i
        self.mapper = LocalPairwiseMap2D(kernel_size)

    def __call__(self, tensor: Tensor) -> Tensor:

        return self.mapper(tensor, self.dist)
