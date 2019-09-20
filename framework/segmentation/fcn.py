from torch import nn
import torch


class FCNSegmentation(nn.Module):

    def __init__(self, output_nc: int, input_nc: int = 3, n_conv: int = 3):
        super(FCNSegmentation, self).__init__()

        layers = [
            nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(output_nc),
        ]

        for i in range(n_conv - 1):
            layers.extend([
                nn.Conv2d(output_nc, output_nc, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(output_nc),
            ])

        layers.extend([
            nn.Conv2d(output_nc, output_nc, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(output_nc),
            nn.Sigmoid()
        ])

        self.net = nn.Sequential(*layers)

    def forward(self, x):

        return self.net(x)
