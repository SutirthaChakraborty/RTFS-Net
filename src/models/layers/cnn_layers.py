import torch
import torch.nn as nn
import torch.nn.functional as F

from . import normalizations, activations


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_chan: int,
        out_chan: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
        padding: int = 0,
        norm_type: str = None,
        act_type: str = None,
        xavier_init: bool = False,
        bias: bool = True,
    ):
        super(ConvNormAct, self).__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.dilation = dilation
        self.padding = padding
        self.norm_type = norm_type
        self.act_type = act_type
        self.xavier_init = xavier_init
        self.bias = bias

        self.conv = nn.Conv1d(
            in_channels=self.in_chan,
            out_channels=self.out_chan,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=self.bias,
        )
        if self.xavier_init:
            nn.init.xavier_uniform_(self.conv.weight)

        self.norm = normalizations.get(self.norm_type)(self.out_chan)
        self.act = activations.get(self.act_type)()

    def forward(self, x):
        output = self.conv(x)
        output = self.norm(output)
        output = self.act(output)
        return output

