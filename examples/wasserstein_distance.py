import torch

n = 1000
xs = (torch.arange(0, n, dtype=torch.float32) / 100.0).view(n, 1)
ys1 = torch.cat((xs.cos(), xs.sin()), dim=1)
ys2 = torch.cat((xs.cos(), xs.sin()), dim=1) * 3

