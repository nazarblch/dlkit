import torch
import torch.nn as nn

from framework.segmentation.unet import double_conv


class Up2xConv2d(nn.Module):
    '''(conv => BN => ReLU)'''

    def __init__(self, in_ch, out_ch):
        super(Up2xConv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.utils.spectral_norm(nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False)),
            double_conv(out_ch, out_ch)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Up4xConv2d(nn.Module):
    '''(conv => BN => ReLU)'''

    def __init__(self, in_ch, out_ch):
        super(Up4xConv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.utils.spectral_norm(nn.ConvTranspose2d(in_ch, out_ch, 4, 1, 0, bias=False)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x