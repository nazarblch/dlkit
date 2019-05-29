from torchvision import models
from torch import nn
from torch import Tensor


class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        for x in range(0, 12):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: Tensor):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        out = [h_relu1, h_relu2, h_relu3]
        return out
