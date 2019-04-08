import torch


BYTES_IN_GB = 1024 ** 3


def freeze(module):
    for param in module.parameters():
        param.requires_grad = False


def unfreeze(module):
    for param in module.parameters():
        param.requires_grad = True


def cuda_memory_use():
    return 'ALLOCATED: {:>6.3f} ({:>6.3f})  CACHED: {:>6.3f} ({:>6.3f})'.format(
        torch.cuda.memory_allocated() / BYTES_IN_GB,
        torch.cuda.max_memory_allocated() / BYTES_IN_GB,
        torch.cuda.memory_cached() / BYTES_IN_GB,
        torch.cuda.max_memory_cached() / BYTES_IN_GB,
    )


def num_params(module):
    return sum(p.numel() for p in module.parameters())

