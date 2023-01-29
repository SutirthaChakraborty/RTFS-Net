import torch
import torch.nn as nn

EPS = torch.finfo(torch.float32).eps


class GlobalLayerNorm(nn.Module):
    def __init__(self, num_channels: int = 1, num_groups: int = 1, eps: float = EPS):
        super(GlobalLayerNorm, self).__init__()
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.eps = eps

        self.norm = nn.GroupNorm(num_groups=self.num_groups, num_channels=self.num_channels, eps=self.eps)

    def forward(self, x):
        return self.norm(x)


gLN = GlobalLayerNorm


def get(identifier):
    if identifier is None:
        return nn.Identity
    elif callable(identifier):
        return identifier
    elif isinstance(identifier, str):
        if hasattr(nn, identifier):
            cls = getattr(nn, identifier)
        else:
            cls = globals().get(identifier)            
        if cls is None:
            raise ValueError("Could not interpret normalization identifier: " + str(identifier))
        return cls
    else:
        raise ValueError("Could not interpret normalization identifier: " + str(identifier))
