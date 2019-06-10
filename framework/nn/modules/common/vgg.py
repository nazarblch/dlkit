from torchvision import models
from torch import nn
from torch import Tensor


class Vgg16(nn.Module):
    def __init__(self, depth: int, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice = nn.Sequential()
        for x in range(0, depth):
            self.slice.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: Tensor):
        return self.slice(x)
