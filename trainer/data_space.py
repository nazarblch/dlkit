import torch
import numpy as np


class DataSpace:
    def __init__(self, name: str, shape: torch.Size):
        self.name = name
        self.shape = shape

