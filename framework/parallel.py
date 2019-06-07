import torch
from typing import List


class ParallelConfig:

    GPU_IDS: List[int] = [0, 1]
    MAIN_DEVICE: torch.device = torch.device("cuda:" + str(GPU_IDS[0]))

