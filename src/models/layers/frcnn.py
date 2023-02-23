import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn_layers import ConvNormAct
from .rnn_layers import TAC


class FRCNNBlock(nn.Module):
    def __init__(
        self,
        in_chan: int,
        hid_chan: int,
        kernel_size: int = 5,
        stride: int = 2,
        norm_type: str = "BatchNorm1d",
        act_type: str = "PReLU",
        upsampling_depth: int = 4,
        dropout: int = -1,
        group_size: int = 1,
    ):
        super(FRCNNBlock, self).__init__()
        self.in_chan = in_chan
        self.hid_chan = hid_chan
        self.kernel_size = kernel_size
        self.stride = stride
        self.norm_type = norm_type
        self.act_type = act_type
        self.upsampling_depth = upsampling_depth
        self.dropout = dropout
        self.group_size = group_size

        self.projection = ConvNormAct(
            in_chan=self.in_chan // self.group_size,
            out_chan=self.hid_chan // self.group_size,
            kernel_size=1,
            norm_type=self.norm_type,
            act_type=self.act_type,
        )
        self.downsample_layers = self.__build_downsample_layers()
        self.fusion_layers = self.__build_fusion_layers()
        self.concat_layers = self.__build_concat_layers()
        self.residual_conv = nn.Sequential(
            ConvNormAct(
                self.hid_chan * self.upsampling_depth // self.group_size,
                self.hid_chan // self.group_size,
                1,
                norm_type=self.norm_type,
                act_type=self.act_type,
            ),
            nn.Conv1d(self.hid_chan // self.group_size, self.in_chan // self.group_size, 1),
        )
        self.dropout_layer = nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity()

    def __build_downsample_layers(self):
        out = nn.ModuleList()
        for i in range(self.upsampling_depth):
            stride = 1 if i == 0 else self.stride
            out.append(
                ConvNormAct(
                    in_chan=self.hid_chan // self.group_size,
                    out_chan=self.hid_chan // self.group_size,
                    kernel_size=self.kernel_size,
                    stride=stride,
                    groups=self.hid_chan // self.group_size,
                    padding=(self.kernel_size - 1) // 2,
                    norm_type=self.norm_type,
                )
            )
        return out

    def __build_fusion_layers(self):
        out = nn.ModuleList()
        for i in range(self.upsampling_depth):
            fuse_layer = nn.ModuleList()
            for j in range(self.upsampling_depth):
                if i == j or (j - i == 1):
                    fuse_layer.append(None)
                elif i - j == 1:
                    fuse_layer.append(
                        ConvNormAct(
                            in_chan=self.hid_chan // self.group_size,
                            out_chan=self.hid_chan // self.group_size,
                            kernel_size=self.kernel_size,
                            stride=self.stride,
                            groups=self.hid_chan // self.group_size,
                            padding=(self.kernel_size - 1) // 2,
                            norm_type=self.norm_type,
                        )
                    )
            out.append(fuse_layer)
        return out

    def __build_concat_layers(self):
        out = nn.ModuleList()
        for i in range(self.upsampling_depth):
            if i == 0 or i == self.upsampling_depth - 1:
                out.append(
                    ConvNormAct(
                        in_chan=self.hid_chan * 2 // self.group_size,
                        out_chan=self.hid_chan // self.group_size,
                        kernel_size=1,
                        norm_type=self.norm_type,
                        act_type=self.act_type,
                    )
                )
            else:
                out.append(
                    ConvNormAct(
                        in_chan=self.hid_chan * 3 // self.group_size,
                        out_chan=self.hid_chan // self.group_size,
                        kernel_size=1,
                        norm_type=self.norm_type,
                        act_type=self.act_type,
                    )
                )
        return out

    def forward(self, x):
        # x: shape (B, C, T)
        residual = x
        x_enc = self.projection(x)

        # bottom-up
        downsampled_outputs = [self.downsample_layers[0](x_enc)]
        for i in range(1, self.upsampling_depth):
            out_i = self.downsample_layers[i](downsampled_outputs[-1])
            downsampled_outputs.append(out_i)

        # lateral connection
        x_fuse = []
        for i in range(self.upsampling_depth):
            shape = downsampled_outputs[i].shape[-1]
            y = torch.cat(
                (
                    self.fusion_layers[i][0](downsampled_outputs[i - 1]) if i - 1 >= 0 else torch.Tensor().to(x_enc.device),
                    downsampled_outputs[i],
                    F.interpolate(downsampled_outputs[i + 1], size=shape, mode="nearest")
                    if i + 1 < self.upsampling_depth
                    else torch.Tensor().to(x_enc.device),
                ),
                dim=1,
            )
            x_fuse.append(self.concat_layers[i](y))

        # resize to T
        shape = downsampled_outputs[0].shape[-1]
        for i in range(1, len(x_fuse)):
            x_fuse[i] = F.interpolate(x_fuse[i], size=shape, mode="nearest")

        # concat and shortcut
        out = self.residual_conv(torch.cat(x_fuse, dim=1))
        # dropout
        out = self.dropout_layer(out) + residual

        return out


class FRCNN(nn.Module):
    def __init__(
        self,
        in_chan: int,
        hid_chan: int,
        kernel_size: int = 5,
        stride: int = 2,
        norm_type: str = "BatchNorm1d",
        act_type: str = "PReLU",
        upsampling_depth: int = 4,
        repeats: int = 4,
        shared: bool = False,
        dropout: float = -1,
        group_size: int = 1,
        *args,
        **kwargs,
    ):
        super(FRCNN, self).__init__()
        self.in_chan = in_chan
        self.hid_chan = hid_chan
        self.kernel_size = kernel_size
        self.stride = stride
        self.norm_type = norm_type
        self.act_type = act_type
        self.upsampling_depth = upsampling_depth
        self.repeats = repeats
        self.shared = shared
        self.dropout = dropout
        self.group_size = group_size

        self.blocks = self.__build_frcnn()
        self.concat_block = self.__build_concat_block()
        self.tac = TAC(self.in_chan // self.group_size, self.hid_chan // self.group_size) if self.group_size > 1 else nn.Identity()

    def __build_frcnn(self):
        if self.shared:
            out = FRCNNBlock(
                in_chan=self.in_chan,
                hid_chan=self.hid_chan,
                kernel_size=self.kernel_size,
                stride=self.stride,
                norm_type=self.norm_type,
                act_type=self.act_type,
                upsampling_depth=self.upsampling_depth,
                dropout=self.dropout,
                group_size=self.group_size,
            )
        else:
            out = nn.ModuleList()
            for _ in range(self.repeats):
                out.append(
                    FRCNNBlock(
                        in_chan=self.in_chan,
                        hid_chan=self.hid_chan,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        norm_type=self.norm_type,
                        act_type=self.act_type,
                        upsampling_depth=self.upsampling_depth,
                        dropout=self.dropout,
                        group_size=self.group_size,
                    )
                )

        return out

    def __build_concat_block(self):
        if self.shared:
            out = ConvNormAct(
                in_chan=self.in_chan // self.group_size,
                out_chan=self.in_chan // self.group_size,
                kernel_size=1,
                groups=self.in_chan // self.group_size,
                act_type=self.act_type,
            )
        else:
            out = nn.ModuleList([None])
            for _ in range(self.repeats - 1):
                out.append(
                    ConvNormAct(
                        in_chan=self.in_chan // self.group_size,
                        out_chan=self.in_chan // self.group_size,
                        kernel_size=1,
                        groups=self.in_chan // self.group_size,
                        act_type=self.act_type,
                    )
                )

        return out

    def get_block(self, i):
        if self.shared:
            return self.blocks
        else:
            return self.blocks[i]

    def get_concat_block(self, i):
        if self.shared:
            return self.concat_block
        else:
            return self.concat_block[i]

    def forward(self, x):
        # x: shape (B, C, T)
        batch_size, _, T = x.shape
        x = self.tac(x.view(batch_size, self.group_size, -1, T)).view(batch_size * self.group_size, -1, T)

        res = x
        for i in range(self.repeats):
            frcnn = self.get_block(i)
            concat_block = self.get_concat_block(i)
            if i == 0:
                x = frcnn(x)
            else:
                x = frcnn(concat_block(res + x))

        return x.view(batch_size, -1, T)
