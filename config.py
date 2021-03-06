import torch
from typing import List


class ParallelConfig:

    GPU_IDS: List[int] = [2,3]
    MAIN_DEVICE: torch.device = torch.device("cuda:" + str(GPU_IDS[0]))

