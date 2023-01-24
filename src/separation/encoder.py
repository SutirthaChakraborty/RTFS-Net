import math
import torch
import inspect
import torch.nn as nn
import torch.nn.functional as F

from ..layers import activations


class Base_Encoder(nn.Module):
    def forward(self, *args, **kwargs):
        raise NotImplementedError


class Convolutional_Encoder(Base_Encoder):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        bias: bool = False,
        encoder_activation: str = None,
    ):
        super(Convolutional_Encoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.encoder_activation = encoder_activation or "linear"

        self.conv = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=self.bias,
        )
        torch.nn.init.xavier_uniform_(self.conv.weight)

        self.encoder = nn.Sequential(self.conv, activations.get(self.encoder_activation)())

    def forward(self, x):
        feature_map = self.encoder(x)

        return feature_map

    def get_config(self):
        encoder_args = {}

        for k, v in (self.__dict__).items():
            if not k.startswith("_") and k != "training":
                if not inspect.ismethod(v):
                    encoder_args[k] = v

        return encoder_args


class Padded_Encoder(nn.Module):
    def __init__(
        self,
        encoder: Base_Encoder,
        in_chan: int,
        upsampling_depth: int = 4,
        kernel_size: int = 21,
    ):
        super(Padded_Encoder, self).__init__()
        self.encoder = encoder
        self.in_chan = in_chan
        self.upsampling_depth = upsampling_depth
        self.kernel_size = kernel_size

        # Appropriate padding is needed for arbitrary lengths
        self.lcm_1 = abs(self.in_chan // 2 * 2**self.upsampling_depth) // math.gcd(self.kernel_size // 2, 2**self.upsampling_depth)
        self.lcm_2 = abs(self.kernel_size // 2 * 2**self.upsampling_depth) // math.gcd(self.kernel_size // 2, 2**self.upsampling_depth)

    def unsqueeze_to_3D(self, x):
        if x.ndim == 1:
            return x.reshape(1, 1, -1)
        elif x.ndim == 2:
            return x.unsqueeze(1)
        else:
            return x

    def pad(self, x, lcm: int):
        values_to_pad = int(x.shape[-1]) % lcm
        if values_to_pad:
            appropriate_shape = x.shape
            padding = torch.zeros(
                list(appropriate_shape[:-1]) + [lcm - values_to_pad],
                dtype=x.dtype,
                device=x.device,
            )
            padded_x = torch.cat([x, padding], dim=-1)
            return padded_x
        else:
            return x

    def forward(self, x):
        x = self.unsqueeze_to_3D(x)

        padded_x = self.pad(x, self.lcm_1)
        padded_x = self.pad(padded_x, self.lcm_2)

        feature_map = self.encoder(padded_x)

        return feature_map

    def get_config(self):
        return self.encoder.get_config()

