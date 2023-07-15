import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import ConvNormAct, InjectionMultiSum
from .gridnet import GridNetBlock


class TDAVNetBlock(nn.Module):
    def __init__(
        self,
        in_chan: int,
        hid_chan: int,
        kernel_size: int = 5,
        stride: int = 2,
        norm_type: str = "gLN",
        act_type: str = "PReLU",
        upsampling_depth: int = 4,
        rnn_1_conf: dict = dict(),
        rnn_2_conf: dict = dict(),
        attention_conf: dict = dict(),
    ):
        super(TDAVNetBlock, self).__init__()
        self.in_chan = in_chan
        self.hid_chan = hid_chan
        self.kernel_size = kernel_size
        self.stride = stride
        self.norm_type = norm_type
        self.act_type = act_type
        self.upsampling_depth = upsampling_depth
        self.rnn_1_conf = rnn_1_conf
        self.rnn_2_conf = rnn_2_conf
        self.attention_conf = attention_conf

        self.downsample_layers = self.__build_downsample_layers()
        self.globalatt = GridNetBlock(self.hid_chan, self.rnn_1_conf, self.rnn_2_conf, self.attention_conf)
        self.fusion_layers = self.__build_fusion_layers()
        self.concat_layers = self.__build_concat_layers()

    def __build_downsample_layers(self):
        out = nn.ModuleList()
        for i in range(self.upsampling_depth):
            out.append(
                ConvNormAct(
                    in_chan=self.hid_chan,
                    out_chan=self.hid_chan,
                    kernel_size=self.kernel_size,
                    stride=1 if i == 0 else self.stride,
                    groups=self.hid_chan,
                    norm_type=self.norm_type,
                    is2d=True,
                )
            )

        return out

    def __build_fusion_layers(self):
        out = nn.ModuleList([])
        for _ in range(self.upsampling_depth):
            out.append(
                InjectionMultiSum(
                    in_chan=self.hid_chan,
                    hid_chan=self.hid_chan,
                    kernel_size=self.kernel_size,
                    norm_type=self.norm_type,
                    is2d=True,
                )
            )

        return out

    def __build_concat_layers(self):
        out = nn.ModuleList([])
        for _ in range(self.upsampling_depth - 1):
            out.append(
                InjectionMultiSum(
                    in_chan=self.hid_chan,
                    hid_chan=self.hid_chan,
                    kernel_size=1,
                    norm_type=self.norm_type,
                    is2d=True,
                )
            )

        return out

    def forward(self, x):
        # x: B, C, T, (F)

        # bottom-up
        downsampled_outputs = [self.downsample_layers[0](x)]
        for i in range(1, self.upsampling_depth):
            downsampled_outputs.append(self.downsample_layers[i](downsampled_outputs[-1]))

        # global pooling
        shape = downsampled_outputs[-1].shape
        global_features = sum(F.adaptive_avg_pool2d(features, output_size=shape[-2:]) for features in downsampled_outputs)

        # global attention module
        global_features = self.globalatt(global_features)  # B, N, T, (F)

        # Gather them now in reverse order
        x_fused = [self.fusion_layers[i](downsampled_outputs[i], global_features) for i in range(self.upsampling_depth)]

        # fuse them into a single vector
        expanded = self.concat_layers[-1](x_fused[-2], x_fused[-1]) + downsampled_outputs[-2]
        for i in range(self.upsampling_depth - 3, -1, -1):
            expanded = self.concat_layers[i](x_fused[i], expanded) + downsampled_outputs[i]

        return expanded


class TDAVNet(nn.Module):
    def __init__(
        self,
        in_chan: int = -1,
        hid_chan: int = -1,
        kernel_size: int = 5,
        stride: int = 2,
        norm_type: str = "gLN",
        act_type: str = "PReLU",
        upsampling_depth: int = 4,
        rnn_1_conf: dict = dict(),
        rnn_2_conf: dict = dict(),
        attention_conf: dict = dict(),
        repeats: int = 4,
        shared: bool = False,
        concat_first: bool = False,
        *args,
        **kwargs,
    ):
        super(TDAVNet, self).__init__()
        self.in_chan = in_chan
        self.hid_chan = hid_chan
        self.kernel_size = kernel_size
        self.stride = stride
        self.norm_type = norm_type
        self.act_type = act_type
        self.upsampling_depth = upsampling_depth
        self.rnn_1_conf = rnn_1_conf
        self.rnn_2_conf = rnn_2_conf
        self.attention_conf = attention_conf
        self.repeats = repeats
        self.shared = shared
        self.concat_first = concat_first

        self.blocks = self.__build_blocks()
        self.concat_block = self.__build_concat_block()

    def __build_blocks(self):
        clss = TDAVNetBlock if (self.in_chan > 0 and self.hid_chan > 0) else nn.Identity
        if self.shared:
            out = clss(
                in_chan=self.in_chan,
                hid_chan=self.hid_chan,
                kernel_size=self.kernel_size,
                stride=self.stride,
                norm_type=self.norm_type,
                act_type=self.act_type,
                upsampling_depth=self.upsampling_depth,
                rnn_1_conf=self.rnn_1_conf,
                rnn_2_conf=self.rnn_2_conf,
                attention_conf=self.attention_conf,
            )
        else:
            out = nn.ModuleList()
            for _ in range(self.repeats):
                out.append(
                    clss(
                        in_chan=self.in_chan,
                        hid_chan=self.hid_chan,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        norm_type=self.norm_type,
                        act_type=self.act_type,
                        upsampling_depth=self.upsampling_depth,
                        rnn_1_conf=self.rnn_1_conf,
                        rnn_2_conf=self.rnn_2_conf,
                        attention_conf=self.attention_conf,
                    )
                )

        return out

    def __build_concat_block(self):
        clss = ConvNormAct if (self.in_chan > 0) and ((self.repeats > 1) or self.concat_first) else nn.Identity
        if self.shared:
            out = clss(
                in_chan=self.in_chan,
                out_chan=self.in_chan,
                kernel_size=1,
                groups=self.in_chan,
                act_type=self.act_type,
                is2d=True,
            )
        else:
            out = nn.ModuleList() if self.concat_first else nn.ModuleList([None])
            for _ in range(self.repeats) if self.concat_first else range(self.repeats - 1):
                out.append(
                    clss(
                        in_chan=self.in_chan,
                        out_chan=self.in_chan,
                        kernel_size=1,
                        groups=self.in_chan,
                        act_type=self.act_type,
                        is2d=True,
                    )
                )

        return out

    def get_block(self, i: int):
        if self.shared:
            return self.blocks
        else:
            return self.blocks[i]

    def get_concat_block(self, i: int):
        if self.shared:
            return self.concat_block
        else:
            return self.concat_block[i]

    def forward(self, x: torch.Tensor):
        residual = x
        for i in range(self.repeats):
            x = self.get_concat_block(i)(x + residual) if i > 0 else x
            x = self.get_block(i)(x)
        return x
