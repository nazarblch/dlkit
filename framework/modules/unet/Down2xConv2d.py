import torch
import torch.nn as nn


class Down2xConv2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down2xConv2d, self).__init__()
        self.mpconv = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x