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

    def forward(self, x: torch.Tensor):
        return self.norm(x)


class LayerNormalization4D(nn.Module):
    def __init__(self, input_dimension, eps: float = EPS):
        super(LayerNormalization4D, self).__init__()
        param_size = [1, input_dimension, 1, 1]
        self.gamma = nn.Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = nn.Parameter(torch.Tensor(*param_size).to(torch.float32))
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x: torch.Tensor):
        mu_ = x.mean(dim=(1,), keepdim=True)  # [B,1,T,F]
        std_ = torch.sqrt(x.var(dim=(1,), unbiased=False, keepdim=True) + self.eps)  # [B,1,T,F]
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat


class LayerNormalizationChanFreq(nn.Module):
    def __init__(self, input_dimension, eps: float = EPS):
        super(LayerNormalizationChanFreq, self).__init__()
        assert len(input_dimension) == 2
        param_size = [1, input_dimension[0], 1, input_dimension[1]]
        self.gamma = nn.Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = nn.Parameter(torch.Tensor(*param_size).to(torch.float32))
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x: torch.Tensor):
        mu_ = x.mean(dim=(1, 3), keepdim=True)  # [B,1,T,1]
        std_ = torch.sqrt(x.var(dim=(1, 3), unbiased=False, keepdim=True) + self.eps)  # [B,1,T,F]
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat


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
