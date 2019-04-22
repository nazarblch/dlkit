import torch


class Module(torch.Module):

    def freeze(module):
        for param in module.parameters():
            param.requires_grad = False

    def unfreeze(module):
        for param in module.parameters():
            param.requires_grad = True

    def num_params(module):
        return sum(p.numel() for p in module.parameters())